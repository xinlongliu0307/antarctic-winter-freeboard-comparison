# Antarctic Winter Freeboard Product Spread

Code and derived outputs for:

> Liu, X., Tilling, R. L., Corney, S. P., Fraser, A. D., and Heil, P. (2026).
> **Antarctic Sea-Ice Freeboard from Envisat and CryoSat-2: Attributing
> Inter-Product Spread to Snow and Radar-Retrieval Contributions.**
> Manuscript under review at *The Cryosphere*.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19977864.svg)](https://doi.org/10.5281/zenodo.19977864)
[![Made with Python](https://img.shields.io/badge/Made%20with-Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)

A preprint of an earlier version of this work is available on ESS Open Archive:
[doi:10.22541/essoar.175795179.91131388/v1](https://doi.org/10.22541/essoar.175795179.91131388/v1).

A preprint of a brief communication version of this work is available on EGUsphere:
[doi:10.5194/egusphere-2026-662](https://doi.org/10.5194/egusphere-2026-662).

---

## Repository scope

This repository provides the processing and figure-generation code, together
with selected derived outputs, used in the study. Specifically, it contains:

- Python scripts that inspect, regrid, mask, sector-average, and intercompare
  Antarctic radar-altimetry freeboard products on the Climate Change
  Initiative (CCI) 50 km EASE2 grid.
- Bootstrap uncertainty quantification and a paired-snow harmonisation
  workflow that isolates the snow and radar-retrieval contributions to the
  inter-product spread.
- MATLAB scripts that generate the compact two-sector time-evolution and
  metrics-dashboard figures from sector-average NetCDF files.
- Derived sector-mean tables, harmonisation tables, alternative
  snow-reference outputs, and common masks / sector definitions referenced
  in the manuscript.

This repository does **not** redistribute the original satellite product
files. Users must obtain those products directly from the original data
providers (see "Data products used" below).

---

## Repository structure

```
antarctic-winter-freeboard-comparison/
├── data/                       # Common masks, sector definitions, derived tables
├── scripts/                    # Python + MATLAB processing and figure scripts
├── .gitignore
├── CITATION.cff                # Machine-readable citation metadata
├── LICENSE                     # MIT License
├── README.md                   # This file
├── environment.yml             # Conda environment specification
└── requirements.txt            # Pip requirement pins
```

---

## Data products used

The workflow ingests four published Antarctic radar-altimetry freeboard
products:

| Product | Mission(s) | Provider |
|---|---|---|
| ESA CCI sea-ice thickness / freeboard | Envisat, CryoSat-2 | European Space Agency Climate Change Initiative |
| LEGOS Antarctic sea-ice thickness / freeboard | Envisat, CryoSat-2 | Laboratoire d'Études en Géophysique et Océanographie Spatiales |
| CryoSat+ Antarctic Ocean (CSAO) | CryoSat-2 | ESA CryoSat+ Antarctic Ocean project |
| CryoSat ThEMatic PrOducts (Cryo-TEMPO) | CryoSat-2 | ESA CryoSat ThEMatic PrOducts |

Auxiliary fields (sea-ice concentration from NSIDC / OSI-SAF and snow-depth
references) are used as described in the manuscript.

---

## Workflow

The Python pipeline is run in numerical order. Each step writes
intermediate outputs to `data/` for use by the following step.

| Step | Script | Purpose |
|---|---|---|
| 00 | `00_inspect_raw_data.py` | Inspect raw product files, verify grids, dimensions, and time coverage |
| 01 | `01_regrid_to_cci_grid.py` | Regrid every product to the CCI 50 km EASE2 grid |
| 02 | `02_common_mask_and_sectors.py` | Build the common winter mask and Weddell / Ross / pan-Antarctic sector masks |
| 03 | `03_snow_harmonisation.py` | Apply paired-snow harmonisation across products |
| 04 | `04_sector_means.py` | Compute monthly and multi-year sector-mean freeboards and write the harmonised NetCDFs |
| 05 | `05_bootstrap_uncertainty.py` | Block-bootstrap uncertainty for sector means and inter-product spread |
| 06 | `06_alternative_snow_reference.py` | Repeat the harmonisation against an alternative snow reference |
| 07 | `07_figures_with_uncertainty.py` | Produce the main manuscript figures with uncertainty envelopes |

The MATLAB figure scripts (`make_ms1_Fig2_TwoSector_TimeEvolution.m` and
`make_ms1_Fig3_MetricsDashboard.m`, with the helper `inspect_ms1_sector_nc.m`)
read the sector-mean NetCDF files produced at step 04 and generate the
compact two-sector and metrics-dashboard figures in the manuscript.

---

## Reproducing the derived outputs

Create the conda environment, activate it, and run the steps in order:

```bash
conda env create -f environment.yml
conda activate ms1-freeboard
python scripts/00_inspect_raw_data.py
python scripts/01_regrid_to_cci_grid.py
python scripts/02_common_mask_and_sectors.py
python scripts/03_snow_harmonisation.py
python scripts/04_sector_means.py
python scripts/05_bootstrap_uncertainty.py
python scripts/06_alternative_snow_reference.py
python scripts/07_figures_with_uncertainty.py
```

If you prefer `pip` over `conda`, an equivalent (Python ≥ 3.10) environment
can be constructed from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The MATLAB figure scripts have been tested on MATLAB R2023a and later.

---

## Software environment

- Python ≥ 3.10
- Core scientific stack: `numpy`, `pandas`, `xarray`, `netCDF4`, `scipy`,
  `matplotlib`, `cartopy`, `pyproj`, `cmocean`
- Regridding: `xesmf` (or a `pyresample`-based fallback)
- Bootstrap / statistics: `scipy.stats`
- MATLAB ≥ R2023a for the figure-generation scripts

The exact pins used to produce the manuscript outputs are recorded in
`environment.yml` and `requirements.txt`.

---

## Data availability

The original ESA CCI, LEGOS, CSAO, and Cryo-TEMPO products are publicly
available from their respective providers and are not redistributed in this
repository. The derived sector-mean tables, harmonisation tables, alternative
snow-reference outputs, and a small set of common masks and sector
definitions sufficient to regenerate the manuscript figures are deposited
together with the code at:

> Liu, X., Tilling, R. L., Corney, S. P., Fraser, A. D., and Heil, P. (2026).
> *Python Scripts and Results for "Antarctic Sea-Ice Freeboard from
> Envisat and CryoSat-2: Attributing Inter-Product Spread to Snow and
> Radar-Retrieval Contributions."* Zenodo.
> https://doi.org/10.5281/zenodo.19977864

---

## How to cite

If you use this code or the derived outputs, please cite both the
forthcoming paper (or the preprint, while the paper is under review) **and**
the archived release. A `CITATION.cff` file is provided in the repository
root for machine-readable citation:

**Paper / preprint**

> Liu, X., Tilling, R. L., Corney, S. P., Fraser, A. D., and Heil, P. (2026).
> Antarctic Sea-Ice Freeboard from Envisat and CryoSat-2: Attributing
> Inter-Product Spread to Snow and Radar-Retrieval Contributions.
> *The Cryosphere*, manuscript under review.
> Preprint: https://doi.org/10.22541/essoar.175795179.91131388/v1

**Software / archived release**

> Liu, X., Tilling, R. L., Corney, S. P., Fraser, A. D., and Heil, P. (2026).
> *Python Scripts and Results for "Antarctic Sea-Ice Freeboard from
> Envisat and CryoSat-2: Attributing Inter-Product Spread to Snow and
> Radar-Retrieval Contributions"* (v1.0.0) [Software]. Zenodo.
> https://doi.org/10.5281/zenodo.19977864

---

## License

This repository is released under the [MIT License](LICENSE).

---

## Contact

Xinlong Liu  
Institute for Marine and Antarctic Studies, University of Tasmania  
xinlong.liu@utas.edu.au
