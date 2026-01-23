function make_ms1_Fig2_TwoSector_TimeEvolution()
%MAKE_MS1_FIG2_TWOSECTOR_TIMEEVOLUTION
% Figure 2 (Main): Two-sector time evolution (Western Weddell + Ross)
% 2 columns: Western Weddell | Ross
% 2 rows: radar freeboard (h_fr) | sea-ice freeboard (h_fi)
% Annual winter means (May-Oct) for each year, 2013–2018.
%
% Output: JPG at 600 dpi.

% -------------------- Paths --------------------
rootDir = 'C:\Users\xliu38\OneDrive - University of Tasmania\PhD_manuscripts\manuscript1';
outDir  = fullfile(rootDir, 'figs_briefcomm');
if ~exist(outDir,'dir'); mkdir(outDir); end

files = struct( ...
    'CCI',       fullfile(rootDir,'cci_cs2_origin_mon_rfb_ifb_sec_avgs.nc'), ...
    'CryoTEMPO', fullfile(rootDir,'cryo_cs2_origin_mon_rfb_ifb_sec_avgs.nc'), ...
    'CSAO',      fullfile(rootDir,'csao_cs2_origin_mon_rfb_ifb_sec_avgs.nc'), ...
    'LEGOS2',    fullfile(rootDir,'legos2_cs2_origin_mon_rfb_ifb_sec_avgs.nc') );

% -------------------- Figure design choices --------------------
yearsTarget  = (2013:2018).';
winterMonths = 5:10; % May--Oct

% Use cell arrays of char for maximum MATLAB compatibility
prodOrder = {'CCI','LEGOS II','CSAO','Cryo-TEMPO'};
prodKey   = {'CCI','LEGOS2','CSAO','CryoTEMPO'}; % struct keys in 'files'

% -------------------- Variable names (char, not string) --------------------
V = struct();

% Western Weddell
V.CCI.hfr.WW       = 'Western Weddell_radar_freeboard';
V.CCI.hfi.WW       = 'Western Weddell_sea_ice_freeboard';

V.CryoTEMPO.hfr.WW = 'Western_Weddell_radar_freeboard';
V.CryoTEMPO.hfi.WW = 'Western_Weddell_sea_ice_freeboard';

V.CSAO.hfr.WW      = 'Western Weddell_radar_freeboard';
V.CSAO.hfi.WW      = 'Western Weddell_sea_ice_freeboard';

V.LEGOS2.hfr.WW    = 'Western Weddell_freeboard_radar';
V.LEGOS2.hfi.WW    = 'Western Weddell_freeboard_ice';

% Ross
V.CCI.hfr.RO       = 'Ross_radar_freeboard';
V.CCI.hfi.RO       = 'Ross_sea_ice_freeboard';

V.CryoTEMPO.hfr.RO = 'Ross_radar_freeboard';
V.CryoTEMPO.hfi.RO = 'Ross_sea_ice_freeboard';

V.CSAO.hfr.RO      = 'Ross_radar_freeboard';
V.CSAO.hfi.RO      = 'Ross_sea_ice_freeboard';

V.LEGOS2.hfr.RO    = 'Ross_freeboard_radar';
V.LEGOS2.hfi.RO    = 'Ross_freeboard_ice';

TIMEVAR = 'time';

% -------------------- Load + compute annual winter means --------------------
S = struct(); % S.(prodKey).hfr_WW etc.

for k = 1:numel(prodKey)
    pk = prodKey{k};  % char
    f  = files.(pk);

    assert(exist(f,'file')==2, 'Missing file: %s', f);

    t  = read_time_as_datetime(f, TIMEVAR);

    % Read monthly sector series (36x1)
    hfr_WW = double(ncread(f, V.(pk).hfr.WW)); hfr_WW = hfr_WW(:);
    hfi_WW = double(ncread(f, V.(pk).hfi.WW)); hfi_WW = hfi_WW(:);
    hfr_RO = double(ncread(f, V.(pk).hfr.RO)); hfr_RO = hfr_RO(:);
    hfi_RO = double(ncread(f, V.(pk).hfi.RO)); hfi_RO = hfi_RO(:);

    S.(pk).years  = yearsTarget;
    S.(pk).hfr_WW = annual_winter_mean(t, hfr_WW, yearsTarget, winterMonths);
    S.(pk).hfi_WW = annual_winter_mean(t, hfi_WW, yearsTarget, winterMonths);
    S.(pk).hfr_RO = annual_winter_mean(t, hfr_RO, yearsTarget, winterMonths);
    S.(pk).hfi_RO = annual_winter_mean(t, hfi_RO, yearsTarget, winterMonths);
end

% -------------------- Plot layout (2x2) --------------------
fig = figure('Color','w','Position',[100 100 1150 650]);
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

% (1) WW hfr
ax1 = nexttile; hold(ax1,'on'); box(ax1,'on'); grid(ax1,'on');
plot_series(ax1, yearsTarget, S, prodKey, 'hfr_WW');
title(ax1, 'Western Weddell');
ylabel(ax1, 'Winter-mean h_{fr} (m)');
xlim(ax1, [min(yearsTarget) max(yearsTarget)]);
xticks(ax1, yearsTarget);

% (2) Ross hfr
ax2 = nexttile; hold(ax2,'on'); box(ax2,'on'); grid(ax2,'on');
plot_series(ax2, yearsTarget, S, prodKey, 'hfr_RO');
title(ax2, 'Ross');
ylabel(ax2, 'Winter-mean h_{fr} (m)');
xlim(ax2, [min(yearsTarget) max(yearsTarget)]);
xticks(ax2, yearsTarget);

% (3) WW hfi
ax3 = nexttile; hold(ax3,'on'); box(ax3,'on'); grid(ax3,'on');
plot_series(ax3, yearsTarget, S, prodKey, 'hfi_WW');
ylabel(ax3, 'Winter-mean h_{fi} (m)');
xlabel(ax3, 'Year');
xlim(ax3, [min(yearsTarget) max(yearsTarget)]);
xticks(ax3, yearsTarget);

% (4) Ross hfi
ax4 = nexttile; hold(ax4,'on'); box(ax4,'on'); grid(ax4,'on');
plot_series(ax4, yearsTarget, S, prodKey, 'hfi_RO');
ylabel(ax4, 'Winter-mean h_{fi} (m)');
xlabel(ax4, 'Year');
xlim(ax4, [min(yearsTarget) max(yearsTarget)]);
xticks(ax4, yearsTarget);

% Legend once
lgd = legend(ax2, prodOrder, 'Location','eastoutside');
lgd.Box = 'off';

% Optional: align y-limits per row
sync_ylim([ax1 ax2]);
sync_ylim([ax3 ax4]);

% -------------------- Export JPG at 600 dpi --------------------
outFile = fullfile(outDir, 'Fig2_TwoSector_TimeEvolution_AnnualWinterMean_2013_2018.jpg');
exportgraphics(fig, outFile, 'Resolution', 600);
fprintf('Saved Figure 2 (600 dpi JPG):\n  %s\n', outFile);

end

% ===================== Helper functions =====================

function plot_series(ax, yrs, S, prodKey, fieldName)
for i = 1:numel(prodKey)
    pk = prodKey{i}; % char
    y  = S.(pk).(fieldName);
    plot(ax, yrs, y, '-o', 'LineWidth', 1.6, 'MarkerSize', 6);
end
end

function y = annual_winter_mean(t, x, yearsTarget, winterMonths)
y = nan(numel(yearsTarget),1);
for i = 1:numel(yearsTarget)
    yy = yearsTarget(i);
    msk = (year(t)==yy) & ismember(month(t), winterMonths);
    if any(msk)
        y(i) = mean(x(msk), 'omitnan');
    end
end
end

function sync_ylim(axs)
mn = +inf; mx = -inf;
for i = 1:numel(axs)
    yl = ylim(axs(i));
    mn = min(mn, yl(1));
    mx = max(mx, yl(2));
end
for i = 1:numel(axs)
    ylim(axs(i), [mn mx]);
end
end

function dt = read_time_as_datetime(ncfile, tvar)
raw = ncread(ncfile, tvar);

% If already datetime-like encoded as char arrays (rare), handle quickly
if ischar(raw)
    dt = datetime(string(raw));
    return
end
if isstring(raw)
    dt = datetime(raw);
    return
end

raw = double(raw(:));

units = '';
try
    units = ncreadatt(ncfile, tvar, 'units');
    if isstring(units); units = char(units); end
catch
    units = '';
end

if isempty(units)
    dt = datetime(raw, 'ConvertFrom','datenum');
    return
end

tokens = regexp(units, '(?<unit>\w+)\s+since\s+(?<ref>[\d\-:T\s]+)', 'names', 'once');
if isempty(tokens)
    dt = datetime(raw, 'ConvertFrom','datenum');
    return
end

refStr = strtrim(tokens.ref);
refStr = strrep(refStr,'T',' ');
refStr = regexprep(refStr,'\s+',' ');

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
    case {'day','days'}
        dt = ref + days(raw);
    case {'hour','hours'}
        dt = ref + hours(raw);
    case {'minute','minutes'}
        dt = ref + minutes(raw);
    case {'second','seconds','sec','secs'}
        dt = ref + seconds(raw);
    case {'month','months'}
        dt = ref + calmonths(raw);
    case {'year','years'}
        dt = ref + calyears(raw);
    otherwise
        dt = ref + days(raw);
end
end
