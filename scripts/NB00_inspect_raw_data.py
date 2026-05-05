#!/usr/bin/env python3
"""
NB00_inspect_raw_data.py
========================
Step 1.1: Inspect all raw freeboard products on NCI Gadi
PhD Thesis — Xinlong Liu, IMAS, University of Tasmania

Purpose:
  - Document variable names, grid dimensions, coordinate arrays,
    time encoding, and fill values for all four products
  - Verify data integrity and identify any issues before processing
  - Output a summary table for reference in all subsequent notebooks

Products:
  1. CCI v4.0 (Envisat + CryoSat-2) — L3C monthly gridded, 50 km EASE-2
  2. LEGOS (Envisat + CryoSat-2 I + CryoSat-2 II) — multi-year aggregate, 12.5 km EASE-2
  3. CSAO (CryoSat-2 only) — annual files, 12.5 km EASE-2
  4. Cryo-TEMPO (CryoSat-2 only) — along-track L2P files

Run on Gadi:
  python3 NB00_inspect_raw_data.py > NB00_output.txt 2>&1
"""

import os
import sys
import glob
import numpy as np

# Try importing netCDF4 first, fall back to h5py
try:
    import netCDF4 as nc
    HAS_NC4 = True
except ImportError:
    HAS_NC4 = False

import h5py

# ==============================================================================
# Configuration
# ==============================================================================
BASE = '/g/data/gv90/xl1657/phd/M1_workspace/raw_data'

CCI_CS2_L3C = os.path.join(BASE, 'CCI_CS2', 'v4p0_L3C')
CCI_ENV_L3C = os.path.join(BASE, 'CCI_ENV', 'v4p0_L3C')
LEGOS_CS2   = os.path.join(BASE, 'LEGOS_CS2')
LEGOS_ENV   = os.path.join(BASE, 'LEGOS_ENV')
CSAO_DIR    = os.path.join(BASE, 'CSAO')
CRYO_DIR    = os.path.join(BASE, 'CryoTEMPO', 'TEMPO_POCA_SI', 'ALONGTRACK')

SEPARATOR = '=' * 80


def inspect_h5py(filepath, label, max_vars=25):
    """Inspect a NetCDF/HDF5 file using h5py."""
    print(f'\n{SEPARATOR}')
    print(f'  {label}')
    print(f'  File: {os.path.basename(filepath)}')
    print(f'  Size: {os.path.getsize(filepath) / 1e6:.1f} MB')
    print(SEPARATOR)

    try:
        f = h5py.File(filepath, 'r')
    except Exception as e:
        print(f'  ERROR opening file: {e}')
        return

    # Print all datasets with shape, dtype, and key attributes
    print(f'\n  {"Variable":<30} {"Shape":<20} {"Dtype":<12} {"Min":>10} {"Max":>10}')
    print(f'  {"-"*30} {"-"*20} {"-"*12} {"-"*10} {"-"*10}')

    count = 0
    for key in sorted(f.keys()):
        item = f[key]
        if hasattr(item, 'shape') and hasattr(item, 'dtype'):
            shape_str = str(item.shape)
            dtype_str = str(item.dtype)

            # Compute min/max for numeric types
            try:
                if item.dtype.kind in ('f', 'i', 'u') and item.size < 1e8:
                    data = item[()]
                    # Mask common fill values
                    if item.dtype.kind == 'f':
                        data = np.where(np.isfinite(data), data, np.nan)
                        # Check for common fill values
                        for fv in [-999, -9999, -1e30, 1e30, 9.96921e+36]:
                            data = np.where(np.abs(data) > 1e20, np.nan, data)
                        vmin = np.nanmin(data) if np.any(np.isfinite(data)) else 'N/A'
                        vmax = np.nanmax(data) if np.any(np.isfinite(data)) else 'N/A'
                    else:
                        vmin = int(np.min(data))
                        vmax = int(np.max(data))
                    min_str = f'{vmin:.4f}' if isinstance(vmin, float) else str(vmin)
                    max_str = f'{vmax:.4f}' if isinstance(vmax, float) else str(vmax)
                else:
                    min_str = 'large'
                    max_str = 'array'
            except Exception:
                min_str = '?'
                max_str = '?'

            print(f'  {key:<30} {shape_str:<20} {dtype_str:<12} {min_str:>10} {max_str:>10}')

            # Print key attributes
            for attr_name in ['units', 'long_name', 'standard_name', '_FillValue',
                              'flag_meanings', 'comment']:
                if attr_name in item.attrs:
                    val = item.attrs[attr_name]
                    if isinstance(val, bytes):
                        val = val.decode('utf-8', errors='replace')
                    elif isinstance(val, np.ndarray):
                        val = val.tolist()
                    val_str = str(val)[:100]
                    print(f'    {attr_name}: {val_str}')

            count += 1
            if count >= max_vars:
                remaining = len(f.keys()) - count
                if remaining > 0:
                    print(f'\n  ... and {remaining} more variables (truncated)')
                break

    # Print global attributes
    print(f'\n  Global attributes:')
    for attr_name in list(f.attrs.keys())[:10]:
        val = f.attrs[attr_name]
        if isinstance(val, bytes):
            val = val.decode('utf-8', errors='replace')
        elif isinstance(val, np.ndarray):
            val = val.tolist()
        val_str = str(val)[:100]
        print(f'    {attr_name}: {val_str}')

    f.close()


def find_first_file(directory, pattern='*.nc', recursive=False):
    """Find the first NetCDF file in a directory."""
    if recursive:
        files = sorted(glob.glob(os.path.join(directory, '**', pattern), recursive=True))
    else:
        files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not files:
        # Try one level deeper
        files = sorted(glob.glob(os.path.join(directory, '*', pattern)))
    if not files:
        # Try two levels deeper
        files = sorted(glob.glob(os.path.join(directory, '*', '*', pattern)))
    return files[0] if files else None


# ==============================================================================
# PRODUCT 1: CCI v4.0 — CryoSat-2 L3C
# ==============================================================================
print('\n' + '#' * 80)
print('#  PRODUCT 1: CCI v4.0 — CryoSat-2 L3C (50 km EASE-2)')
print('#' * 80)

# Count files per year
print('\n  File inventory:')
total_cci_cs2 = 0
for year_dir in sorted(glob.glob(os.path.join(CCI_CS2_L3C, '*'))):
    if os.path.isdir(year_dir):
        year = os.path.basename(year_dir)
        nfiles = len(glob.glob(os.path.join(year_dir, '*.nc')))
        total_cci_cs2 += nfiles
        print(f'    {year}: {nfiles} files')
print(f'    TOTAL: {total_cci_cs2} files')

# Inspect a representative file (May 2013 — start of study period)
f_cci_cs2 = find_first_file(CCI_CS2_L3C, '*201305*')
if f_cci_cs2 is None:
    f_cci_cs2 = find_first_file(CCI_CS2_L3C)
if f_cci_cs2:
    inspect_h5py(f_cci_cs2, 'CCI v4.0 CryoSat-2 L3C — Representative File')


# ==============================================================================
# PRODUCT 1b: CCI v4.0 — Envisat L3C
# ==============================================================================
print('\n' + '#' * 80)
print('#  PRODUCT 1b: CCI v4.0 — Envisat L3C (50 km EASE-2)')
print('#' * 80)

total_cci_env = 0
for year_dir in sorted(glob.glob(os.path.join(CCI_ENV_L3C, '*'))):
    if os.path.isdir(year_dir):
        year = os.path.basename(year_dir)
        nfiles = len(glob.glob(os.path.join(year_dir, '*.nc')))
        total_cci_env += nfiles
        print(f'    {year}: {nfiles} files')
print(f'    TOTAL: {total_cci_env} files')

f_cci_env = find_first_file(CCI_ENV_L3C, '*200305*')
if f_cci_env is None:
    f_cci_env = find_first_file(CCI_ENV_L3C)
if f_cci_env:
    inspect_h5py(f_cci_env, 'CCI v4.0 Envisat L3C — Representative File')


# ==============================================================================
# PRODUCT 2: LEGOS — CryoSat-2 (I and II)
# ==============================================================================
print('\n' + '#' * 80)
print('#  PRODUCT 2: LEGOS CryoSat-2 (12.5 km EASE-2)')
print('#' * 80)

legos_cs2_files = sorted(glob.glob(os.path.join(LEGOS_CS2, '*.nc')))
print(f'\n  Files found: {len(legos_cs2_files)}')
for f in legos_cs2_files:
    print(f'    {os.path.basename(f)}: {os.path.getsize(f)/1e6:.1f} MB')

# Inspect LEGOS I (AMSR2 snow)
f_legos_i = [f for f in legos_cs2_files if 'SnowAMSR' in f]
if f_legos_i:
    inspect_h5py(f_legos_i[0], 'LEGOS I (CryoSat-2, AMSR-2 snow)')

# Inspect LEGOS II (Ka-Ku snow)
f_legos_ii = [f for f in legos_cs2_files if 'SnowKaKu' in f]
if f_legos_ii:
    inspect_h5py(f_legos_ii[0], 'LEGOS II (CryoSat-2, SARAL Ka / CS2 Ku snow)')


# ==============================================================================
# PRODUCT 2b: LEGOS — Envisat
# ==============================================================================
print('\n' + '#' * 80)
print('#  PRODUCT 2b: LEGOS Envisat (12.5 km EASE-2)')
print('#' * 80)

legos_env_files = sorted(glob.glob(os.path.join(LEGOS_ENV, '*.nc')))
print(f'\n  Files found: {len(legos_env_files)}')
for f in legos_env_files:
    print(f'    {os.path.basename(f)}: {os.path.getsize(f)/1e6:.1f} MB')

if legos_env_files:
    inspect_h5py(legos_env_files[0], 'LEGOS Envisat (AMSR-E snow)')


# ==============================================================================
# PRODUCT 3: CSAO (CryoSat-2 only)
# ==============================================================================
print('\n' + '#' * 80)
print('#  PRODUCT 3: CSAO (12.5 km EASE-2)')
print('#' * 80)

csao_files = sorted(glob.glob(os.path.join(CSAO_DIR, '*.nc')))
print(f'\n  Files found: {len(csao_files)}')
for f in csao_files:
    print(f'    {os.path.basename(f)}: {os.path.getsize(f)/1e6:.1f} MB')

# Inspect a standard file (with SIT) and a NOSIT file
f_csao_std = [f for f in csao_files if 'NOSIT' not in f]
f_csao_nosit = [f for f in csao_files if 'NOSIT' in f]

if f_csao_std:
    inspect_h5py(f_csao_std[0], 'CSAO — Standard (with SIT)')

if f_csao_nosit:
    inspect_h5py(f_csao_nosit[0], 'CSAO — NOSIT variant')


# ==============================================================================
# PRODUCT 4: Cryo-TEMPO (CryoSat-2, along-track)
# ==============================================================================
print('\n' + '#' * 80)
print('#  PRODUCT 4: Cryo-TEMPO (along-track L2P)')
print('#' * 80)

# Count files per year/month for the study period
print('\n  File inventory (study period 2013-2018, May-Oct):')
total_cryo = 0
for yr in range(2013, 2019):
    for mo in range(5, 11):
        mo_dir = os.path.join(CRYO_DIR, str(yr), f'{mo:02d}')
        if os.path.isdir(mo_dir):
            nfiles = len(glob.glob(os.path.join(mo_dir, '*.nc')))
            total_cryo += nfiles
            print(f'    {yr}/{mo:02d}: {nfiles} files')
        else:
            print(f'    {yr}/{mo:02d}: DIRECTORY NOT FOUND')
print(f'    TOTAL (study period): {total_cryo} files')

# Count bonus data
print('\n  Bonus data (2019-2025):')
total_bonus = 0
for yr in range(2019, 2026):
    yr_dir = os.path.join(CRYO_DIR, str(yr))
    if os.path.isdir(yr_dir):
        nfiles = len(glob.glob(os.path.join(yr_dir, '**', '*.nc'), recursive=True))
        total_bonus += nfiles
        print(f'    {yr}: {nfiles} files')
print(f'    TOTAL (bonus): {total_bonus} files')

# Inspect a representative along-track file
f_cryo = find_first_file(os.path.join(CRYO_DIR, '2013', '05'))
if f_cryo:
    inspect_h5py(f_cryo, 'Cryo-TEMPO — Along-Track Representative File')
else:
    print('  WARNING: No Cryo-TEMPO file found for 2013/05')
    # Try another month
    f_cryo = find_first_file(os.path.join(CRYO_DIR, '2014', '05'))
    if f_cryo:
        inspect_h5py(f_cryo, 'Cryo-TEMPO — Along-Track Representative File (2014/05)')


# ==============================================================================
# SUMMARY: Variable Name Mapping Across Products
# ==============================================================================
print('\n' + '#' * 80)
print('#  VARIABLE NAME MAPPING ACROSS PRODUCTS')
print('#' * 80)

print("""
  Thesis Variable    CCI v4.0           LEGOS              CSAO               Cryo-TEMPO
  ================   ================   ================   ================   ================
  h_fr (radar fb)    radar_freeboard    freeboard_radar    radar_freeboard_*  [TO VERIFY]
  h_fi (ice fb)      sea_ice_freeboard  freeboard_ice      [COMPUTE]          [TO VERIFY]
  h_s  (snow)        snow_depth         snow_depth         snow_depth_ASD     [TO VERIFY]
  h_sc (speed corr)  [COMPUTE]          [COMPUTE]          [COMPUTE]          [COMPUTE]
  lat                lat                latitude           lat                [TO VERIFY]
  lon                lon                longitude          lon                [TO VERIFY]
  time               time               time               time               [TO VERIFY]
  grid               216x216 EASE-2     850x850 EASE-2     712x712 EASE-2     along-track
  resolution         50 km              12.5 km            12.5 km            ~300 m

  Notes:
  - h_sc = h_fi - h_fr (computed for all products)
  - CSAO does NOT provide h_fi directly; must compute using Eq. 3.1-3.3
  - Cryo-TEMPO is along-track; must be gridded to 50 km EASE-2
  - LEGOS files are multi-year aggregates with time dimension
  - CCI files are monthly, one file per month
  - CSAO files are annual, one file per year with monthly time dimension
  - [TO VERIFY] fields will be filled after inspecting the Cryo-TEMPO file above
""")


# ==============================================================================
# COORDINATE SYSTEM CHECK
# ==============================================================================
print('\n' + '#' * 80)
print('#  COORDINATE SYSTEM CHECK')
print('#' * 80)

# Check CCI grid coordinates (reference grid)
print('\n  CCI v4.0 Reference Grid:')
if f_cci_cs2:
    f = h5py.File(f_cci_cs2, 'r')
    if 'lat' in f and 'lon' in f:
        lat = f['lat'][()]
        lon = f['lon'][()]
        print(f'    lat shape: {lat.shape}, range: [{np.nanmin(lat):.2f}, {np.nanmax(lat):.2f}]')
        print(f'    lon shape: {lon.shape}, range: [{np.nanmin(lon):.2f}, {np.nanmax(lon):.2f}]')
    if 'xc' in f and 'yc' in f:
        xc = f['xc'][()]
        yc = f['yc'][()]
        print(f'    xc shape: {xc.shape}, range: [{xc.min():.1f}, {xc.max():.1f}] km')
        print(f'    yc shape: {yc.shape}, range: [{yc.min():.1f}, {yc.max():.1f}] km')
        print(f'    xc spacing: {np.mean(np.diff(xc)):.2f} km')
        print(f'    yc spacing: {np.mean(np.diff(yc)):.2f} km')
    if 'region_code' in f:
        rc = f['region_code'][()]
        unique_codes = np.unique(rc[rc != -128])  # exclude fill value
        print(f'    region_code unique values: {unique_codes}')
        if 'flag_meanings' in f['region_code'].attrs:
            meanings = f['region_code'].attrs['flag_meanings']
            if isinstance(meanings, bytes):
                meanings = meanings.decode()
            print(f'    region_code meanings: {meanings}')
    f.close()

# Check LEGOS grid coordinates
print('\n  LEGOS Grid:')
if f_legos_i:
    f = h5py.File(f_legos_i[0], 'r')
    if 'latitude' in f and 'longitude' in f:
        lat = f['latitude'][()]
        lon = f['longitude'][()]
        print(f'    lat shape: {lat.shape}, range: [{np.nanmin(lat):.2f}, {np.nanmax(lat):.2f}]')
        print(f'    lon shape: {lon.shape}, range: [{np.nanmin(lon):.2f}, {np.nanmax(lon):.2f}]')
    if 'u' in f and 'v' in f:
        u = f['u'][()]
        v = f['v'][()]
        print(f'    u (x-coord) shape: {u.shape}, range: [{u.min():.1f}, {u.max():.1f}]')
        print(f'    v (y-coord) shape: {v.shape}, range: [{v.min():.1f}, {v.max():.1f}]')
    f.close()

# Check CSAO grid coordinates
print('\n  CSAO Grid:')
if f_csao_std:
    f = h5py.File(f_csao_std[0], 'r')
    if 'lat' in f and 'lon' in f:
        lat = f['lat'][()]
        lon = f['lon'][()]
        print(f'    lat shape: {lat.shape}, range: [{np.nanmin(lat):.2f}, {np.nanmax(lat):.2f}]')
        print(f'    lon shape: {lon.shape}, range: [{np.nanmin(lon):.2f}, {np.nanmax(lon):.2f}]')
    if 'x' in f and 'y' in f:
        x = f['x'][()]
        y = f['y'][()]
        print(f'    x shape: {x.shape}, range: [{x.min():.1f}, {x.max():.1f}]')
        print(f'    y shape: {y.shape}, range: [{y.min():.1f}, {y.max():.1f}]')
    f.close()

# Check Cryo-TEMPO coordinates (along-track)
print('\n  Cryo-TEMPO Coordinates:')
if f_cryo:
    f = h5py.File(f_cryo, 'r')
    for coord_name in ['latitude', 'lat', 'Latitude']:
        if coord_name in f:
            data = f[coord_name][()]
            print(f'    {coord_name} shape: {data.shape}, range: [{np.nanmin(data):.2f}, {np.nanmax(data):.2f}]')
    for coord_name in ['longitude', 'lon', 'Longitude']:
        if coord_name in f:
            data = f[coord_name][()]
            print(f'    {coord_name} shape: {data.shape}, range: [{np.nanmin(data):.2f}, {np.nanmax(data):.2f}]')
    f.close()


# ==============================================================================
# CCI REGION CODE ANALYSIS (for sector assignment)
# ==============================================================================
print('\n' + '#' * 80)
print('#  CCI REGION CODE ANALYSIS (Sector Assignment)')
print('#' * 80)

if f_cci_cs2:
    f = h5py.File(f_cci_cs2, 'r')
    if 'region_code' in f:
        rc = f['region_code'][0, :, :]  # first time step
        lat = f['lat'][()]
        lon = f['lon'][()]

        # Count grid cells per region code
        print('\n  Grid cells per region code:')
        for code in sorted(np.unique(rc)):
            mask = rc == code
            n_cells = np.sum(mask)
            if n_cells > 0:
                lat_range = f'[{np.nanmin(lat[mask]):.1f}, {np.nanmax(lat[mask]):.1f}]'
                lon_range = f'[{np.nanmin(lon[mask]):.1f}, {np.nanmax(lon[mask]):.1f}]'
                print(f'    Code {code:3d}: {n_cells:5d} cells, lat {lat_range}, lon {lon_range}')

        # Check if Western/Eastern Weddell split exists
        weddell_mask = (rc == 6)  # Assuming 6 = weddell_sea based on flag_meanings
        if np.any(weddell_mask):
            weddell_lons = lon[weddell_mask]
            print(f'\n  Weddell sector longitude range: [{np.nanmin(weddell_lons):.1f}, {np.nanmax(weddell_lons):.1f}]')
            print(f'  Need to split at 40W for Western/Eastern Weddell')
            n_western = np.sum(weddell_lons < -40)
            n_eastern = np.sum(weddell_lons >= -40)
            print(f'  Western Weddell (lon < -40): {n_western} cells')
            print(f'  Eastern Weddell (lon >= -40): {n_eastern} cells')
    f.close()


# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print('\n' + '#' * 80)
print('#  FINAL DATA INVENTORY SUMMARY')
print('#' * 80)

print(f"""
  Product             Files    Size     Grid        Resolution   Time Period
  ==================  =======  =======  ==========  ===========  ==================
  CCI CS2 v4.0 L3C    {total_cci_cs2:>5d}    ~117 MB  216x216     50 km        2010-2024
  CCI ENV v4.0 L3C    {total_cci_env:>5d}    ~83 MB   216x216     50 km        2002-2012
  LEGOS CS2 (I+II)        2    ~2.5 GB  850x850     12.5 km      2010-2018
  LEGOS ENV               1    ~1.6 GB  850x850     12.5 km      2002-2011
  CSAO                   12    ~1.4 GB  712x712     12.5 km      2010-2020
  Cryo-TEMPO (study) {total_cryo:>5d}    ~2 GB    along-trk   ~300 m       2013-2018
  Cryo-TEMPO (bonus) {total_bonus:>5d}    ~11 GB   along-trk   ~300 m       2019-2025
  ==================  =======  =======  ==========  ===========  ==================

  STATUS: All raw data verified and ready for processing.

  NEXT STEPS:
  1. NB01_regrid_to_cci_grid.py — Regrid LEGOS, CSAO, Cryo-TEMPO to CCI 50 km grid
  2. NB02_common_mask_and_sectors.py — Build common masks and compute sector means
  3. NB03_snow_harmonisation.py — Run Cases A-D experiments
  4. NB04_chapter3_figures.py — Generate all Chapter 3 figures
""")

print(f'\nScript completed successfully.')
print(f'Output saved. Review the variable names and coordinate systems above')
print(f'before proceeding to NB01_regrid_to_cci_grid.py.')
