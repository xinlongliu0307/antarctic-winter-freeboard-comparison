function make_ms1_Fig3_MetricsDashboard()
%MAKE_MS1_FIG3_METRICSDASHBOARD
% Figure 3 (Main): Combined metrics dashboard relative to CCI
% Sectors: Western Weddell, Ross
% Products (vs CCI): LEGOS II, CSAO, Cryo-TEMPO
% Variables: h_fr (radar freeboard), h_fi (sea-ice freeboard)
% Period: 2013–2018, winter months May–Oct (monthly points pooled)
%
% Output: JPG at 600 dpi (and optionally PDF vector)

% -------------------- Paths --------------------
rootDir = 'C:\Users\xliu38\OneDrive - University of Tasmania\PhD_manuscripts\manuscript1';
outDir  = fullfile(rootDir, 'figs_briefcomm');
if ~exist(outDir,'dir'); mkdir(outDir); end

files = struct( ...
    'CCI',       fullfile(rootDir,'cci_cs2_origin_mon_rfb_ifb_sec_avgs.nc'), ...
    'CryoTEMPO', fullfile(rootDir,'cryo_cs2_origin_mon_rfb_ifb_sec_avgs.nc'), ...
    'CSAO',      fullfile(rootDir,'csao_cs2_origin_mon_rfb_ifb_sec_avgs.nc'), ...
    'LEGOS2',    fullfile(rootDir,'legos2_cs2_origin_mon_rfb_ifb_sec_avgs.nc') );

% -------------------- Settings --------------------
yearsTarget  = 2013:2018;
winterMonths = 5:10; % May--Oct
TIMEVAR      = 'time';

% We only show these three products (relative to CCI)
prodKey   = {'LEGOS2','CSAO','CryoTEMPO'};            % struct keys in "files"
prodLabel = {'LEGOS II','CSAO','Cryo-TEMPO'};         % for plotting

% Sectors (two-story sectors)
secKey    = {'WW','RO'};
secLabel  = {'Western Weddell','Ross'};

% Variables (two rows)
varKey    = {'hfr','hfi'};
varLabel  = {'$h_{fr}$','$h_{fi}$'};

% -------------------- Variable name mapping (char) --------------------
V = struct();

% CCI
V.CCI.hfr.WW = 'Western Weddell_radar_freeboard';
V.CCI.hfi.WW = 'Western Weddell_sea_ice_freeboard';
V.CCI.hfr.RO = 'Ross_radar_freeboard';
V.CCI.hfi.RO = 'Ross_sea_ice_freeboard';

% Cryo-TEMPO
V.CryoTEMPO.hfr.WW = 'Western_Weddell_radar_freeboard';
V.CryoTEMPO.hfi.WW = 'Western_Weddell_sea_ice_freeboard';
V.CryoTEMPO.hfr.RO = 'Ross_radar_freeboard';
V.CryoTEMPO.hfi.RO = 'Ross_sea_ice_freeboard';

% CSAO
V.CSAO.hfr.WW = 'Western Weddell_radar_freeboard';
V.CSAO.hfi.WW = 'Western Weddell_sea_ice_freeboard';
V.CSAO.hfr.RO = 'Ross_radar_freeboard';
V.CSAO.hfi.RO = 'Ross_sea_ice_freeboard';

% LEGOS II (LEGOS2 file)
V.LEGOS2.hfr.WW = 'Western Weddell_freeboard_radar';
V.LEGOS2.hfi.WW = 'Western Weddell_freeboard_ice';
V.LEGOS2.hfr.RO = 'Ross_freeboard_radar';
V.LEGOS2.hfi.RO = 'Ross_freeboard_ice';

% -------------------- Compute metrics --------------------
% Output matrices: rows = (hfr,hfi), cols = (prod×sector) = 6
nRow = numel(varKey);
nCol = numel(prodKey) * numel(secKey);

R    = nan(nRow, nCol);
BIAS = nan(nRow, nCol);
RMSE = nan(nRow, nCol);

colLabels = cell(1, nCol);
c = 0;

for p = 1:numel(prodKey)
    pk = prodKey{p};
    pl = prodLabel{p};

    for s = 1:numel(secKey)
        sk = secKey{s};
        sl = secLabel{s};

        c = c + 1;
        colLabels{c} = sprintf('%s | %s', pl, sl);

        % Build aligned monthly vectors (winter months pooled, 2013–2018)
        for v = 1:numel(varKey)
            vk = varKey{v};

            % Read product series table
            Tp = read_monthly_series(files.(pk), TIMEVAR, V.(pk).(vk).(sk), yearsTarget, winterMonths);

            % Read CCI series table
            Tc = read_monthly_series(files.CCI, TIMEVAR, V.CCI.(vk).(sk), yearsTarget, winterMonths);

            % Inner join on month-key
            Tj = innerjoin(Tp, Tc, 'Keys','tkey', 'LeftVariables',{'tkey','val'}, 'RightVariables',{'val'});

            xp = Tj.val_Tp;   % product
            xc = Tj.val_Tc;   % CCI

            % Remove NaN pairs
            ok = isfinite(xp) & isfinite(xc);
            xp = xp(ok);
            xc = xc(ok);

            if numel(xp) >= 3
                R(v,c)    = corr(xp, xc);
                d         = xp - xc;
                BIAS(v,c) = mean(d, 'omitnan');
                RMSE(v,c) = sqrt(mean(d.^2, 'omitnan'));
            end
        end
    end
end

% -------------------- Plot dashboard (3 subpanels) --------------------
fig = figure('Color','w','Position',[100 100 1600 520]);
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

% Panel A: r (divergent, centered at 0)
ax1 = nexttile;
plot_heatmap(ax1, R, varLabel, colLabels, '$r$', [-1 1], '%.2f', true);

% Panel B: bias (divergent, centered at 0)
ax2 = nexttile;
bmax = max(abs(BIAS(:)), [], 'omitnan');
if isempty(bmax) || ~isfinite(bmax); bmax = 0.05; end
plot_heatmap(ax2, BIAS, varLabel, colLabels, 'Bias (m)', [-bmax bmax], '%.3f', true);

% Panel C: RMSE (sequential)
ax3 = nexttile;
rmax = max(RMSE(:), [], 'omitnan');
if isempty(rmax) || ~isfinite(rmax); rmax = 0.10; end
plot_heatmap(ax3, RMSE, varLabel, colLabels, 'RMSE (m)', [0 rmax], '%.3f', false);


% -------------------- Export --------------------
outJpg = fullfile(outDir, 'Fig3_MetricsDashboard_WW_Ross_2013_2018_MayOct.jpg');
exportgraphics(fig, outJpg, 'Resolution', 600);
fprintf('Saved Figure 3 (600 dpi JPG):\n  %s\n', outJpg);

% Optional (recommended for LaTeX): vector PDF
outPdf = fullfile(outDir, 'Fig3_MetricsDashboard_WW_Ross_2013_2018_MayOct.pdf');
exportgraphics(fig, outPdf, 'ContentType','vector');
fprintf('Saved Figure 3 (vector PDF):\n  %s\n', outPdf);

end

% ===================== Helpers =====================

function T = read_monthly_series(ncfile, tvar, vname, yearsTarget, winterMonths)
% Returns table with:
%   tkey = datetime(year,month,1)
%   val  = variable value

assert(exist(ncfile,'file')==2, 'Missing file: %s', ncfile);

t  = read_time_as_datetime(ncfile, tvar);
x  = double(ncread(ncfile, vname));
x  = x(:);

% Month key to align products even if timestamps are mid-month
tkey = datetime(year(t), month(t), 1);

% Filter years + winter months
ok = ismember(year(tkey), yearsTarget) & ismember(month(tkey), winterMonths);
tkey = tkey(ok);
x    = x(ok);

% If duplicates exist (unlikely), average them
[ukey, ~, ic] = unique(tkey);
xmean = accumarray(ic, x, [], @(z) mean(z,'omitnan'));

T = table(ukey, xmean, 'VariableNames', {'tkey','val'});
end

function plot_heatmap(ax, M, rowLabels, colLabels, ttl, clim, fmt, divergeZero)
%PLOT_HEATMAP Heatmap with optional divergent colourmap centred at 0.
%
% divergeZero = true  -> symmetric colour limits around 0 and divergent map
% divergeZero = false -> use given clim and sequential map

imagesc(ax, M);
axis(ax, 'tight');

% ---- Tick labels ----
set(ax, 'YTick', 1:numel(rowLabels), 'YTickLabel', rowLabels);
set(ax, 'XTick', 1:numel(colLabels), 'XTickLabel', colLabels, 'XTickLabelRotation', 45);

% LaTeX for h_{fr}, h_{fi} row labels and math titles if provided
set(ax, 'TickLabelInterpreter', 'latex');

% Title interpreter (latex to support $r$)
title(ax, ttl, 'Interpreter','latex');

% ---- Colour limits & colormap ----
if divergeZero
    % Force symmetric limits around 0 (middle = 0)
    if isempty(clim) || numel(clim) ~= 2 || ~all(isfinite(clim))
        mmax = max(abs(M(:)), [], 'omitnan');
        if isempty(mmax) || ~isfinite(mmax); mmax = 1; end
        clim = [-mmax, mmax];
    else
        mmax = max(abs(clim));
        clim = [-mmax, mmax];
    end
    caxis(ax, clim);

    % Divergent colormap: prefer "balance" if available, otherwise fallback
    if exist('balance','file') == 2
        colormap(ax, balance);
    else
        colormap(ax, bluewhitered(256)); % fallback included below
    end
else
    if ~isempty(clim) && numel(clim) == 2 && all(isfinite(clim))
        caxis(ax, clim);
    end
    colormap(ax, parula);
end

cb = colorbar(ax);
cb.Box = 'off';

% ---- Numeric annotations ----
[nr, nc] = size(M);
for i = 1:nr
    for j = 1:nc
        val = M(i,j);
        if isfinite(val)
            text(ax, j, i, sprintf(fmt, val), ...
                'HorizontalAlignment','center', ...
                'VerticalAlignment','middle', ...
                'FontSize', 10, 'Color','k', ...
                'Interpreter','latex'); % safe even for numbers
        end
    end
end

ax.Layer = 'top';
ax.Box = 'on';

end

function cmap = bluewhitered(m)
%BLUEWHITERED Simple divergent colormap, centred at white (0).
% Fallback if "balance" is unavailable.
if nargin < 1; m = 256; end
m = max(3, round(m));

% Build two linear ramps: blue->white and white->red
m1 = floor(m/2);
m2 = m - m1;

blue  = [0 0 1];
white = [1 1 1];
red   = [1 0 0];

c1 = [linspace(blue(1),  white(1), m1)', ...
      linspace(blue(2),  white(2), m1)', ...
      linspace(blue(3),  white(3), m1)'];

c2 = [linspace(white(1), red(1),  m2)', ...
      linspace(white(2), red(2),  m2)', ...
      linspace(white(3), red(3),  m2)'];

cmap = [c1; c2];
end


function dt = read_time_as_datetime(ncfile, tvar)
raw = ncread(ncfile, tvar);

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
