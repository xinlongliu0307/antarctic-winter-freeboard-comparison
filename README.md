# Antarctic winter freeboard product spread: scripts and derived outputs

This repository contains the processing scripts and selected derived outputs used for the study:

**Antarctic Sea-Ice Freeboard from Envisat and CryoSat-2: Attributing Inter-Product Spread to Snow Assumptions and Radar-Retrieval Baselines**, 2026

## Repository scope

This repository provides:
- scripts used to inspect, regrid, mask, sector-average, and compare Antarctic radar-altimetry freeboard products;
- derived sector-mean and harmonisation tables used in the manuscript;
- common masks and sector definitions on the Climate Change Initiative (CCI) 50 km EASE2 grid;
- figure-generation scripts for the main manuscript figures.

This repository does not redistribute the original satellite product files. Users must obtain those products from the original data providers.

## Data products used

The workflow uses:
- European Space Agency (ESA) Envisat and CryoSat-2 CCI sea-ice thickness / freeboard products;
- Laboratoire d'Etudes en G\'{e}ophysique et Oc\'{e}anographie Spatiales (LEGOS) Envisat and CryoSat-2 Antarctic sea-ice thickness / freeboard products;
- CryoSat+ Antarctic Ocean (CSAO) CryoSat-2 Antarctic sea-ice product;
- CryoSat ThEMatic PrOducts (Cryo-TEMPO) CryoSat-2 sea-ice product.

## Workflow

The main Python workflow is:

1. `00_inspect_raw_data.py`
2. `01_regrid_to_cci_grid.py`
3. `02_common_mask_and_sectors.py`
4. `03_snow_harmonisation.py`
5. `05_bootstrap_uncertainty.py`
6. `06_alternative_snow_reference.py`
7. `07_figures_with_uncertainty.py`

The MATLAB scripts generate the compact manuscript figures from sector-average NetCDF files.

## Reproducing the derived outputs

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate ms1-freeboard
