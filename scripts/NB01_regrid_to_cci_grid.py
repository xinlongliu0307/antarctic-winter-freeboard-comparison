#!/usr/bin/env python3
"""
NB01_regrid_to_cci_grid.py
==========================
Step 1.2: Regrid LEGOS, CSAO, and Cryo-TEMPO to the CCI 50 km EASE-2 grid.

PhD Thesis — Xinlong Liu, IMAS, University of Tasmania

Purpose:
  - Extract the CCI 216x216 EASE-2 grid as the reference target grid.
  - Regrid LEGOS (850x850, 12.5 km) → 50 km using pyresample.
  - Regrid CSAO (712x712, 12.5 km) → 50 km using pyresample.
  - Bin Cryo-TEMPO along-track data → monthly 50 km grid cells.
  - Save all regridded products as monthly NetCDF files in processed/regridded/.

Verified variable names (from NB00):
  CCI:        radar_freeboard, sea_ice_freeboard, snow_depth, lat, lon
  LEGOS:      freeboard_radar, freeboard_ice, snow_depth, latitude, longitude
  CSAO:       radar_freeboard_mean, snow_depth_ASD, lat, lon
  Cryo-TEMPO: radar_freeboard, sea_ice_freeboard, snow_depth, latitude, longitude

Run on Gadi:
  python3 NB01_regrid_to_cci_grid.py 2>&1 | tee NB01_output.txt
"""

import os
import sys
import glob
import numpy as np
import h5py
from datetime import datetime, timedelta

# ==============================================================================
# Configuration
# ==============================================================================
BASE_RAW = '/g/data/gv90/xl1657/phd/M1_workspace/raw_data'
BASE_OUT = '/g/data/gv90/xl1657/phd/M1_workspace/processed/regridded'
os.makedirs(BASE_OUT, exist_ok=True)

# Study periods
ENV_YEARS = range(2003, 2012)   # 2003-2011
CS2_YEARS = range(2013, 2019)   # 2013-2018
WINTER_MONTHS = range(5, 11)    # May(5) to October(10)

# Fill value used by LEGOS and CSAO
LEGOS_FILL = -2147483648.0
CSAO_FILL  = -2147483648.0

SEPARATOR = '=' * 70


# ==============================================================================
# STEP 0: Extract CCI reference grid
# ==============================================================================
def load_cci_reference_grid():
    """Load the CCI 216x216 EASE-2 grid coordinates."""
    print(f'\n{SEPARATOR}')
    print('  Loading CCI reference grid...')
    print(SEPARATOR)

    cci_dir = os.path.join(BASE_RAW, 'CCI_CS2', 'v4p0_L3C')
    # Find any CCI file to extract the grid
    cci_file = sorted(glob.glob(os.path.join(cci_dir, '**', '*.nc'), recursive=True))[0]

    f = h5py.File(cci_file, 'r')
    lat_grid = f['lat'][()]          # (216, 216)
    lon_grid = f['lon'][()]          # (216, 216)
    xc = f['xc'][()]                 # (216,) in km
    yc = f['yc'][()]                 # (216,) in km
    f.close()

    print(f'  Grid shape: {lat_grid.shape}')
    print(f'  Lat range: [{lat_grid.min():.2f}, {lat_grid.max():.2f}]')
    print(f'  Lon range: [{lon_grid.min():.2f}, {lon_grid.max():.2f}]')
    print(f'  xc range: [{xc.min():.1f}, {xc.max():.1f}] km, spacing: {np.mean(np.diff(xc)):.1f} km')

    return lat_grid, lon_grid, xc, yc


# ==============================================================================
# STEP 1: Load and save CCI monthly fields (already on target grid)
# ==============================================================================
def process_cci(lat_grid, lon_grid, era='CS2'):
    """
    CCI is already on the reference grid — just extract the relevant
    variables for each month and save in a consistent format.
    """
    print(f'\n{SEPARATOR}')
    print(f'  Processing CCI v4.0 ({era} era)...')
    print(SEPARATOR)

    if era == 'CS2':
        cci_dir = os.path.join(BASE_RAW, 'CCI_CS2', 'v4p0_L3C')
        years = CS2_YEARS
    else:
        cci_dir = os.path.join(BASE_RAW, 'CCI_ENV', 'v4p0_L3C')
        years = ENV_YEARS

    count = 0
    for year in years:
        for month in WINTER_MONTHS:
            # Find the file for this year/month
            pattern = f'*{year}{month:02d}*fv4p0.nc'
            files = glob.glob(os.path.join(cci_dir, str(year), pattern))
            if not files:
                files = glob.glob(os.path.join(cci_dir, '**', pattern), recursive=True)
            if not files:
                print(f'    {year}/{month:02d}: FILE NOT FOUND')
                continue

            f = h5py.File(files[0], 'r')
            hfr = f['radar_freeboard'][0, :, :]
            hfi = f['sea_ice_freeboard'][0, :, :]
            hs  = f['snow_depth'][0, :, :]
            hfr_unc = f['radar_freeboard_uncertainty'][0, :, :]
            hfi_unc = f['sea_ice_freeboard_uncertainty'][0, :, :]
            hs_unc  = f['snow_depth_uncertainty'][0, :, :]
            sic = f['sea_ice_concentration'][0, :, :]
            region = f['region_code'][0, :, :]
            quality = f['quality_flag'][0, :, :]
            status = f['status_flag'][0, :, :]
            f.close()

            # Mask invalid data (large fill values)
            for arr in [hfr, hfi, hs, hfr_unc, hfi_unc, hs_unc, sic]:
                arr[~np.isfinite(arr)] = np.nan

            # Compute speed correction
            hsc = hfi - hfr

            # Save
            outfile = os.path.join(BASE_OUT, f'CCI_{era}_{year}{month:02d}.npz')
            np.savez_compressed(outfile,
                hfr=hfr, hfi=hfi, hs=hs, hsc=hsc,
                hfr_unc=hfr_unc, hfi_unc=hfi_unc, hs_unc=hs_unc,
                sic=sic, region_code=region, quality_flag=quality,
                status_flag=status,
                lat=lat_grid, lon=lon_grid,
                year=year, month=month, product='CCI', era=era)
            count += 1

    print(f'  Saved {count} monthly files for CCI {era}.')


# ==============================================================================
# STEP 2: Regrid LEGOS to CCI grid using nearest-neighbour lookup
# ==============================================================================
def build_regrid_index(source_lat, source_lon, target_lat, target_lon):
    """
    Build a lookup table mapping each target grid cell to the nearest
    source grid cell. Uses a simple brute-force approach with a KD-tree
    built from lat/lon coordinates converted to Cartesian.
    """
    from scipy.spatial import cKDTree

    # Convert lat/lon to Cartesian (unit sphere)
    def latlon_to_xyz(lat, lon):
        lat_r = np.radians(lat)
        lon_r = np.radians(lon)
        x = np.cos(lat_r) * np.cos(lon_r)
        y = np.cos(lat_r) * np.sin(lon_r)
        z = np.sin(lat_r)
        return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)

    source_xyz = latlon_to_xyz(source_lat, source_lon)
    target_xyz = latlon_to_xyz(target_lat, target_lon)

    tree = cKDTree(source_xyz)
    distances, indices = tree.query(target_xyz, k=1)

    # Convert flat indices to 2D indices
    rows = indices // source_lat.shape[1]
    cols = indices % source_lat.shape[1]

    return rows.reshape(target_lat.shape), cols.reshape(target_lat.shape), distances.reshape(target_lat.shape)


def regrid_field(data, row_idx, col_idx, distances, max_dist_km=75.0):
    """
    Regrid a 2D field using precomputed nearest-neighbour indices.
    Mask cells where the nearest source cell is farther than max_dist_km.
    """
    # Earth radius ~ 6371 km; angular distance on unit sphere to km
    max_dist_rad = max_dist_km / 6371.0
    # cKDTree distances are Euclidean on unit sphere ≈ angular distance for small angles
    max_dist_xyz = 2 * np.sin(max_dist_rad / 2)

    regridded = data[row_idx, col_idx].astype(np.float32)
    regridded[distances > max_dist_xyz] = np.nan
    return regridded


def process_legos(target_lat, target_lon):
    """Regrid LEGOS products to the CCI 50 km grid."""
    print(f'\n{SEPARATOR}')
    print('  Processing LEGOS products...')
    print(SEPARATOR)

    # --- LEGOS I (CryoSat-2, AMSR-2 snow) ---
    legos_i_file = os.path.join(BASE_RAW, 'LEGOS_CS2',
                                'SIT_SH_2010_2018_CS2_SnowAMSR.ease2_12500_smth25000.nc')
    f = h5py.File(legos_i_file, 'r')
    legos_lat = f['latitude'][()]    # (850, 850)
    legos_lon = f['longitude'][()]   # (850, 850)
    legos_i_hfr = f['freeboard_radar'][()]     # (49, 850, 850)
    legos_i_hfi = f['freeboard_ice'][()]       # (49, 850, 850)
    legos_i_hs  = f['snow_depth'][()]          # (49, 850, 850)
    legos_i_time = f['time'][()]               # (49,) days since 2000-01-01
    f.close()

    # Mask fill values
    for arr in [legos_i_hfr, legos_i_hfi, legos_i_hs]:
        arr[arr < -1e9] = np.nan

    # Build regridding lookup (same grid for all LEGOS products)
    print('  Building LEGOS -> CCI regridding index...')
    row_idx, col_idx, distances = build_regrid_index(legos_lat, legos_lon, target_lat, target_lon)
    print(f'  Max regridding distance: {distances.max() * 6371:.1f} km')

    # Convert LEGOS time to year/month
    ref_date = datetime(2000, 1, 1)

    count = 0
    for t_idx in range(len(legos_i_time)):
        dt = ref_date + timedelta(days=int(legos_i_time[t_idx]))
        year, month = dt.year, dt.month

        if year not in CS2_YEARS or month not in WINTER_MONTHS:
            continue

        hfr = regrid_field(legos_i_hfr[t_idx], row_idx, col_idx, distances)
        hfi = regrid_field(legos_i_hfi[t_idx], row_idx, col_idx, distances)
        hs  = regrid_field(legos_i_hs[t_idx], row_idx, col_idx, distances)
        hsc = hfi - hfr

        outfile = os.path.join(BASE_OUT, f'LEGOS_I_CS2_{year}{month:02d}.npz')
        np.savez_compressed(outfile,
            hfr=hfr, hfi=hfi, hs=hs, hsc=hsc,
            lat=target_lat, lon=target_lon,
            year=year, month=month, product='LEGOS_I', era='CS2')
        count += 1
        print(f'    LEGOS I: {year}/{month:02d} saved')

    print(f'  Saved {count} monthly files for LEGOS I.')

    # --- LEGOS II (CryoSat-2, Ka-Ku snow) ---
    legos_ii_file = os.path.join(BASE_RAW, 'LEGOS_CS2',
                                 'SIT_SH_2013_2018_CS2_SnowKaKu.ease2_12500_smth25000.nc')
    f = h5py.File(legos_ii_file, 'r')
    legos_ii_hfr = f['freeboard_radar'][()]
    legos_ii_hfi = f['freeboard_ice'][()]
    legos_ii_hs  = f['snow_depth'][()]
    legos_ii_time = f['time'][()]
    f.close()

    for arr in [legos_ii_hfr, legos_ii_hfi, legos_ii_hs]:
        arr[arr < -1e9] = np.nan

    count = 0
    for t_idx in range(len(legos_ii_time)):
        dt = ref_date + timedelta(days=int(legos_ii_time[t_idx]))
        year, month = dt.year, dt.month

        if year not in CS2_YEARS or month not in WINTER_MONTHS:
            continue

        hfr = regrid_field(legos_ii_hfr[t_idx], row_idx, col_idx, distances)
        hfi = regrid_field(legos_ii_hfi[t_idx], row_idx, col_idx, distances)
        hs  = regrid_field(legos_ii_hs[t_idx], row_idx, col_idx, distances)
        hsc = hfi - hfr

        outfile = os.path.join(BASE_OUT, f'LEGOS_II_CS2_{year}{month:02d}.npz')
        np.savez_compressed(outfile,
            hfr=hfr, hfi=hfi, hs=hs, hsc=hsc,
            lat=target_lat, lon=target_lon,
            year=year, month=month, product='LEGOS_II', era='CS2')
        count += 1
        print(f'    LEGOS II: {year}/{month:02d} saved')

    print(f'  Saved {count} monthly files for LEGOS II.')

    # --- LEGOS Envisat ---
    legos_env_file = os.path.join(BASE_RAW, 'LEGOS_ENV',
                                  'SIT_SH_2002_2011_ENV_SnowAMSR.ease2_12500_smth25000.nc')
    f = h5py.File(legos_env_file, 'r')
    legos_env_hfr = f['freeboard_radar'][()]
    legos_env_hfi = f['freeboard_ice'][()]
    legos_env_hs  = f['snow_depth'][()]
    legos_env_time = f['time'][()]
    f.close()

    for arr in [legos_env_hfr, legos_env_hfi, legos_env_hs]:
        arr[arr < -1e9] = np.nan

    count = 0
    for t_idx in range(len(legos_env_time)):
        dt = ref_date + timedelta(days=int(legos_env_time[t_idx]))
        year, month = dt.year, dt.month

        if year not in ENV_YEARS or month not in WINTER_MONTHS:
            continue

        hfr = regrid_field(legos_env_hfr[t_idx], row_idx, col_idx, distances)
        hfi = regrid_field(legos_env_hfi[t_idx], row_idx, col_idx, distances)
        hs  = regrid_field(legos_env_hs[t_idx], row_idx, col_idx, distances)
        hsc = hfi - hfr

        outfile = os.path.join(BASE_OUT, f'LEGOS_ENV_{year}{month:02d}.npz')
        np.savez_compressed(outfile,
            hfr=hfr, hfi=hfi, hs=hs, hsc=hsc,
            lat=target_lat, lon=target_lon,
            year=year, month=month, product='LEGOS', era='ENV')
        count += 1
        print(f'    LEGOS ENV: {year}/{month:02d} saved')

    print(f'  Saved {count} monthly files for LEGOS ENV.')


# ==============================================================================
# STEP 3: Regrid CSAO to CCI grid
# ==============================================================================
def process_csao(target_lat, target_lon):
    """Regrid CSAO to the CCI 50 km grid."""
    print(f'\n{SEPARATOR}')
    print('  Processing CSAO...')
    print(SEPARATOR)

    csao_dir = os.path.join(BASE_RAW, 'CSAO')

    # Load CSAO grid from a standard (non-NOSIT) file
    csao_std_files = sorted([f for f in glob.glob(os.path.join(csao_dir, '*.nc'))
                              if 'NOSIT' not in f])
    if not csao_std_files:
        print('  ERROR: No standard CSAO files found.')
        return

    f = h5py.File(csao_std_files[0], 'r')
    csao_lat = f['lat'][()]    # (712, 712)
    csao_lon = f['lon'][()]    # (712, 712)
    f.close()

    # Build regridding lookup
    print('  Building CSAO -> CCI regridding index...')
    row_idx, col_idx, distances = build_regrid_index(csao_lat, csao_lon, target_lat, target_lon)
    print(f'  Max regridding distance: {distances.max() * 6371:.1f} km')

    # CSAO time: days since 1950-01-01
    csao_ref_date = datetime(1950, 1, 1)

    count = 0
    for csao_file in csao_std_files:
        fname = os.path.basename(csao_file)
        # Extract year from filename: fb_sla_cs2_sam_YYYY.nc
        try:
            file_year = int(fname.split('_')[-1].replace('.nc', ''))
        except ValueError:
            print(f'    Skipping {fname}: cannot parse year')
            continue

        try:
            f = h5py.File(csao_file, 'r')
        except OSError as e:
            print(f'    Skipping {fname}: {e}')
            continue
        # Handle variable name differences across CSAO file versions
        if 'radar_freeboard_mean' in f:
            hfr_all = f['radar_freeboard_mean'][()]
            hs_all  = f['snow_depth_ASD'][()]
        elif 'radar_freeboard_20hz_mean' in f:
            hfr_all = f['radar_freeboard_20hz_mean'][()]
            hs_all  = f['snow_depth_sd_ASD_sh'][()]
        else:
            print(f'    Skipping {fname}: unrecognised variable names')
            f.close()
            continue
        time_arr = f['time'][()]
        f.close()

        # Mask fill values
        for arr in [hfr_all, hs_all]:
            arr[arr < -1e9] = np.nan

        for t_idx in range(len(time_arr)):
            # Convert time to date
            try:
                dt = csao_ref_date + timedelta(days=int(time_arr[t_idx]))
                year, month = dt.year, dt.month
            except (ValueError, OverflowError):
                continue

            if year not in CS2_YEARS or month not in WINTER_MONTHS:
                continue

            hfr = regrid_field(hfr_all[t_idx], row_idx, col_idx, distances)
            hs  = regrid_field(hs_all[t_idx], row_idx, col_idx, distances)

            # CSAO does not provide h_fi — compute using Eq 3.1-3.3
            # Use Kurtz & Markus (2012) density for CSAO
            if month == 5:
                rho_s = 320.0
            elif month == 10:
                rho_s = 340.0
            else:
                rho_s = 350.0

            c = 3e8  # speed of light
            cs = c * (1 + 5.1e-4 * rho_s) ** (-1.5)
            hsc = (c / cs - 1) * hs
            hfi = hfr + hsc

            outfile = os.path.join(BASE_OUT, f'CSAO_CS2_{year}{month:02d}.npz')
            np.savez_compressed(outfile,
                hfr=hfr, hfi=hfi, hs=hs, hsc=hsc,
                lat=target_lat, lon=target_lon,
                year=year, month=month, product='CSAO', era='CS2')
            count += 1
            print(f'    CSAO: {year}/{month:02d} saved')

    print(f'  Saved {count} monthly files for CSAO.')


# ==============================================================================
# STEP 4: Bin Cryo-TEMPO along-track data to monthly 50 km grid
# ==============================================================================
def process_cryotempo(target_lat, target_lon, xc, yc):
    """
    Bin Cryo-TEMPO along-track observations into monthly 50 km grid cells.

    Approach A: For each month, read all along-track files, assign each
    observation to the nearest CCI grid cell, and compute the cell mean.
    """
    print(f'\n{SEPARATOR}')
    print('  Processing Cryo-TEMPO (along-track → monthly 50 km grid)...')
    print(SEPARATOR)

    cryo_dir = os.path.join(BASE_RAW, 'CryoTEMPO', 'TEMPO_POCA_SI', 'ALONGTRACK')

    # Build a KD-tree from CCI grid for fast nearest-neighbour assignment
    from scipy.spatial import cKDTree

    def latlon_to_xyz(lat, lon):
        lat_r = np.radians(lat)
        lon_r = np.radians(lon)
        x = np.cos(lat_r) * np.cos(lon_r)
        y = np.cos(lat_r) * np.sin(lon_r)
        z = np.sin(lat_r)
        return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)

    target_xyz = latlon_to_xyz(target_lat, target_lon)
    tree = cKDTree(target_xyz)

    # Maximum assignment distance: 50 km grid → ~35 km half-diagonal
    max_dist_km = 40.0
    max_dist_xyz = 2 * np.sin((max_dist_km / 6371.0) / 2)

    ny, nx = target_lat.shape  # 216, 216

    count = 0
    for year in CS2_YEARS:
        for month in WINTER_MONTHS:
            mo_dir = os.path.join(cryo_dir, str(year), f'{month:02d}')
            if not os.path.isdir(mo_dir):
                print(f'    Cryo-TEMPO: {year}/{month:02d} — directory not found')
                continue

            nc_files = sorted(glob.glob(os.path.join(mo_dir, '*.nc')))
            if not nc_files:
                print(f'    Cryo-TEMPO: {year}/{month:02d} — no files')
                continue

            # Accumulators for binning
            hfr_sum = np.zeros((ny, nx), dtype=np.float64)
            hfi_sum = np.zeros((ny, nx), dtype=np.float64)
            hs_sum  = np.zeros((ny, nx), dtype=np.float64)
            n_obs   = np.zeros((ny, nx), dtype=np.int32)

            n_files_read = 0
            n_points_total = 0

            for nc_file in nc_files:
                try:
                    f = h5py.File(nc_file, 'r')
                    lat_obs = f['latitude'][()]
                    lon_obs = f['longitude'][()]
                    hfr_obs = f['radar_freeboard'][()]
                    hfi_obs = f['sea_ice_freeboard'][()]
                    hs_obs  = f['snow_depth'][()]
                    f.close()
                except Exception:
                    continue

                # Filter valid observations
                valid = (np.isfinite(hfr_obs) & np.isfinite(hfi_obs) &
                         np.isfinite(hs_obs) & np.isfinite(lat_obs) &
                         (lat_obs < -50))  # Antarctic only

                if not np.any(valid):
                    continue

                lat_v = lat_obs[valid]
                lon_v = lon_obs[valid]
                hfr_v = hfr_obs[valid]
                hfi_v = hfi_obs[valid]
                hs_v  = hs_obs[valid]

                # Assign to nearest CCI grid cell
                obs_xyz = latlon_to_xyz(lat_v, lon_v)
                dists, flat_indices = tree.query(obs_xyz)

                # Convert flat indices to 2D
                row_indices = flat_indices // nx
                col_indices = flat_indices % nx

                # Filter by maximum distance
                close_enough = dists < max_dist_xyz

                for i in range(len(lat_v)):
                    if close_enough[i]:
                        r, c = row_indices[i], col_indices[i]
                        hfr_sum[r, c] += hfr_v[i]
                        hfi_sum[r, c] += hfi_v[i]
                        hs_sum[r, c]  += hs_v[i]
                        n_obs[r, c]   += 1

                n_files_read += 1
                n_points_total += np.sum(close_enough)

            # Compute monthly means
            mask = n_obs > 0
            hfr_mean = np.full((ny, nx), np.nan, dtype=np.float32)
            hfi_mean = np.full((ny, nx), np.nan, dtype=np.float32)
            hs_mean  = np.full((ny, nx), np.nan, dtype=np.float32)

            hfr_mean[mask] = (hfr_sum[mask] / n_obs[mask]).astype(np.float32)
            hfi_mean[mask] = (hfi_sum[mask] / n_obs[mask]).astype(np.float32)
            hs_mean[mask]  = (hs_sum[mask] / n_obs[mask]).astype(np.float32)
            hsc_mean = hfi_mean - hfr_mean

            outfile = os.path.join(BASE_OUT, f'CryoTEMPO_CS2_{year}{month:02d}.npz')
            np.savez_compressed(outfile,
                hfr=hfr_mean, hfi=hfi_mean, hs=hs_mean, hsc=hsc_mean,
                n_obs=n_obs,
                lat=target_lat, lon=target_lon,
                year=year, month=month, product='CryoTEMPO', era='CS2')
            count += 1

            n_valid_cells = np.sum(mask)
            print(f'    Cryo-TEMPO: {year}/{month:02d} — {n_files_read} files, '
                  f'{n_points_total} obs, {n_valid_cells} grid cells with data')

    print(f'  Saved {count} monthly files for Cryo-TEMPO.')


# ==============================================================================
# STEP 5: Verification
# ==============================================================================
def verify_outputs():
    """Count and summarise all regridded files."""
    print(f'\n{SEPARATOR}')
    print('  VERIFICATION: Regridded Output Files')
    print(SEPARATOR)

    products = {
        'CCI_CS2':      'CCI_CS2_*.npz',
        'CCI_ENV':      'CCI_ENV_*.npz',
        'LEGOS_I':      'LEGOS_I_CS2_*.npz',
        'LEGOS_II':     'LEGOS_II_CS2_*.npz',
        'LEGOS_ENV':    'LEGOS_ENV_*.npz',
        'CSAO':         'CSAO_CS2_*.npz',
        'CryoTEMPO':    'CryoTEMPO_CS2_*.npz',
    }

    print(f'\n  {"Product":<15} {"Files":<8} {"Example file":<40}')
    print(f'  {"-"*15} {"-"*8} {"-"*40}')

    total = 0
    for name, pattern in products.items():
        files = sorted(glob.glob(os.path.join(BASE_OUT, pattern)))
        n = len(files)
        total += n
        example = os.path.basename(files[0]) if files else 'NONE'
        print(f'  {name:<15} {n:<8} {example:<40}')

    print(f'\n  Total regridded files: {total}')

    # Spot-check one file
    test_files = glob.glob(os.path.join(BASE_OUT, 'CCI_CS2_201305.npz'))
    if test_files:
        data = np.load(test_files[0])
        print(f'\n  Spot-check CCI_CS2_201305.npz:')
        for key in ['hfr', 'hfi', 'hs', 'hsc']:
            arr = data[key]
            valid = np.isfinite(arr)
            if np.any(valid):
                print(f'    {key}: shape={arr.shape}, '
                      f'valid={np.sum(valid)}, '
                      f'mean={np.nanmean(arr):.4f}, '
                      f'range=[{np.nanmin(arr):.4f}, {np.nanmax(arr):.4f}]')
            else:
                print(f'    {key}: shape={arr.shape}, all NaN')
        data.close()

    print(f'\n  All regridded files saved to: {BASE_OUT}')
    print(f'  Ready for NB02_common_mask_and_sectors.py')


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == '__main__':
    print(f'\nNB01_regrid_to_cci_grid.py')
    print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Output directory: {BASE_OUT}')

    # Load reference grid
    lat_grid, lon_grid, xc, yc = load_cci_reference_grid()

    # Process each product
    process_cci(lat_grid, lon_grid, era='CS2')
    process_cci(lat_grid, lon_grid, era='ENV')
    process_legos(lat_grid, lon_grid)
    process_csao(lat_grid, lon_grid)
    process_cryotempo(lat_grid, lon_grid, xc, yc)

    # Verify
    verify_outputs()

    print(f'\nCompleted: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('Done.')
