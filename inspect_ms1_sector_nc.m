function inspect_ms1_sector_nc()
%INSPECT_MS1_SECTOR_NC
% Inspect MS1 sector-average NetCDF files (CCI / LEGOS2 / CSAO / Cryo-TEMPO)
% and help you confirm variable names for h_fr/h_fi and Western Weddell/Ross.

rootDir = 'C:\Users\xliu38\OneDrive - University of Tasmania\PhD_manuscripts\manuscript1';

files = { ...
    'cci_cs2_origin_mon_rfb_ifb_sec_avgs.nc', ...
    'cryo_cs2_origin_mon_rfb_ifb_sec_avgs.nc', ...
    'csao_cs2_origin_mon_rfb_ifb_sec_avgs.nc', ...
    'legos2_cs2_origin_mon_rfb_ifb_sec_avgs.nc'};

fprintf('\n=== MS1 NetCDF inspection ===\nRoot: %s\n\n', rootDir);

for i = 1:numel(files)
    f = fullfile(rootDir, files{i});
    fprintf('------------------------------------------------------------\n');
    fprintf('[%d/%d] %s\n', i, numel(files), f);

    if exist(f,'file') ~= 2
        fprintf('  ERROR: File not found.\n');
        continue
    end

    info = ncinfo(f);

    % ---- Dimensions ----
    fprintf('\n  Dimensions:\n');
    for d = 1:numel(info.Dimensions)
        dim = info.Dimensions(d);
        fprintf('    - %-20s  %d\n', dim.Name, dim.Length);
    end

    % ---- Variables (name + size) ----
    fprintf('\n  Variables:\n');
    for v = 1:numel(info.Variables)
        vn = info.Variables(v).Name;
        sz = info.Variables(v).Size;
        fprintf('    - %-35s  [%s]\n', vn, strjoin(string(sz), 'x'));
    end

    % ---- Global attributes (key ones) ----
    fprintf('\n  Global attributes (first ~15):\n');
    nAtt = min(15, numel(info.Attributes));
    for a = 1:nAtt
        an = info.Attributes(a).Name;
        av = info.Attributes(a).Value;
        if ischar(av) || isstring(av)
            avs = string(av);
            if strlength(avs) > 80, avs = extractBefore(avs, 80) + "..."; end
            fprintf('    * %s: %s\n', an, avs);
        else
            fprintf('    * %s: [non-text]\n', an);
        end
    end

    % ---- Detect time variable and show range ----
    tvar = pick_time_var(info);
    if tvar == ""
        fprintf('\n  Time variable: NOT detected (will look for year/month variables later).\n');
    else
        try
            t = read_time_as_datetime(f, tvar);
            fprintf('\n  Time variable: %s\n', tvar);
            fprintf('    Time range: %s  to  %s  (n=%d)\n', string(min(t)), string(max(t)), numel(t));
            fprintf('    First 5: %s\n', strjoin(string(t(1:min(5,end))), ', '));
        catch ME
            fprintf('\n  Time variable: %s (detected) but conversion failed: %s\n', tvar, ME.message);
        end
    end

    % ---- Suggest candidates for h_fr / h_fi and sectors ----
    vnames = string({info.Variables.Name});
    fprintf('\n  Candidate variables (by name pattern):\n');

    c_hfr = vnames(contains(lower(vnames), ["hfr","rfb","radar_freeboard","freeboard_radar"]));
    c_hfi = vnames(contains(lower(vnames), ["hfi","ifb","sea_ice_freeboard","freeboard_ice"]));

    c_ww  = vnames(contains(lower(vnames), ["w_wed","west_weddell","western_weddell","weddell_west","wweddell"]));
    c_ro  = vnames(contains(lower(vnames), ["ross"]));

    fprintf('    h_fr-like: %s\n', join_or_none(c_hfr));
    fprintf('    h_fi-like: %s\n', join_or_none(c_hfi));
    fprintf('    sector tag (W Weddell) in name: %s\n', join_or_none(c_ww));
    fprintf('    sector tag (Ross) in name: %s\n', join_or_none(c_ro));

    % ---- Quick peek into likely 1-D sector series variables (optional) ----
    % This tries to identify variables that look like time-series vectors.
    fprintf('\n  Likely time-series vectors (size contains time length):\n');
    tlen = detect_time_length(info, tvar);
    if ~isnan(tlen)
        for v = 1:numel(info.Variables)
            vn = info.Variables(v).Name;
            sz = info.Variables(v).Size;
            if isvector(sz) && any(sz == tlen) && prod(sz) == tlen
                if contains(lower(vn), ["rfb","ifb","hfr","hfi","freeboard"])
                    fprintf('    - %s  [%s]\n', vn, strjoin(string(sz),'x'));
                end
            end
        end
    end

    fprintf('\n');
end

fprintf('=== Inspection complete ===\n');
fprintf('Next: Use the printed candidate names to configure Figure 2 extraction.\n');

end

% -------------------- helpers --------------------

function out = join_or_none(s)
if isempty(s), out = "(none)"; else, out = strjoin(s, ', '); end
end

function tvar = pick_time_var(info)
vnames = lower(string({info.Variables.Name}));
% prefer exact 'time', then common variations
cands = ["time","t","date","datetime","time_counter","time_month","time_year"];
for k = 1:numel(cands)
    idx = find(vnames == cands(k), 1);
    if ~isempty(idx)
        tvar = string(info.Variables(idx).Name);
        return
    end
end
% fallback: any variable containing 'time'
idx = find(contains(vnames,"time"), 1);
if ~isempty(idx)
    tvar = string(info.Variables(idx).Name);
else
    tvar = "";
end
end

function tlen = detect_time_length(info, tvar)
tlen = NaN;
if tvar == "", return; end
% try read variable size from ncinfo
v = info.Variables(strcmp({info.Variables.Name}, char(tvar)));
if ~isempty(v)
    tlen = v.Size(1);
end
end

function dt = read_time_as_datetime(ncfile, tvar)
% Converts NetCDF time to MATLAB datetime where possible.
raw = ncread(ncfile, char(tvar));

% If it's already char/string dates:
if ischar(raw)
    dt = datetime(string(raw));
    return
elseif isstring(raw)
    dt = datetime(raw);
    return
end

% If there are year/month variables instead of CF-time
% (handled elsewhere; here we assume CF-like time)
units = "";
try
    units = string(ncreadatt(ncfile, char(tvar), 'units'));
catch
    units = "";
end

raw = double(raw(:));

if units == ""
    % fallback: interpret as MATLAB datenum-like if values are large
    % (rare; mostly CF units exist). You can adjust if needed.
    dt = datetime(raw, 'ConvertFrom','datenum');
    return
end

% Typical: "days since YYYY-MM-DD ..." / "seconds since ..."
tokens = regexp(units, '(?<unit>\w+)\s+since\s+(?<ref>[\d\-:T\s]+)', 'names', 'once');
if isempty(tokens)
    % Could be "months since", etc.
    dt = datetime(raw, 'ConvertFrom','datenum');
    return
end

refStr = strtrim(tokens.ref);
refStr = strrep(refStr,'T',' ');
refStr = regexprep(refStr,'\s+',' ');

% Try robust reference parsing
try
    ref = datetime(refStr, 'InputFormat','yyyy-MM-dd HH:mm:ss');
catch
    try
        ref = datetime(refStr, 'InputFormat','yyyy-MM-dd');
    catch
        ref = datetime(refStr);
    end
end

u = lower(tokens.unit);
switch u
    case {"day","days"}
        dt = ref + days(raw);
    case {"hour","hours"}
        dt = ref + hours(raw);
    case {"minute","minutes"}
        dt = ref + minutes(raw);
    case {"second","seconds","sec","secs"}
        dt = ref + seconds(raw);
    case {"month","months"}
        dt = ref + calmonths(raw);
    case {"year","years"}
        dt = ref + calyears(raw);
    otherwise
        dt = ref + days(raw);
end
end
