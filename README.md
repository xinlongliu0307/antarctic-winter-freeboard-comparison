# Antarctic Winter Freeboard Product Spread

Code and derived outputs for:

> Liu, X., Tilling, R. L., Corney, S. P., Fraser, A. D., and Heil, P. (2026).
> **Antarctic Sea-Ice Freeboard from Envisat and CryoSat-2: Attributing
> Inter-Product Spread to Snow and Radar-Retrieval Contributions.**
> Manuscript in revision at *The Cryosphere*.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19977864.svg)](https://doi.org/10.5281/zenodo.19977864)
[![Made with Python](https://img.shields.io/badge/Made%20with-Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)

A preprint of a brief communication version of this work is available on EGUsphere:
[doi:10.5194/egusphere-2026-662](https://doi.org/10.5194/egusphere-2026-662).

A preprint of an earlier version of this work is available on ESS Open Archive:
[doi:10.22541/essoar.175795179.91131388/v1](https://doi.org/10.22541/essoar.175795179.91131388/v1)
and [doi:10.22541/essoar.174309801.17791750/v1](https://essopenarchive.org/doi/full/10.22541/essoar.174309801.17791750/v1).

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
├── output/                     # Terminal output of each script
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
| 00 | `NB00_inspect_raw_data.py` | Inspect raw product files; verify grids, dimensions and time coverage |
| 01 | `NB01_regrid_to_cci_grid.py` | Map LEGOS and CSAO onto the CCI 50 km EASE2 grid by nearest-neighbour sampling; bin along-track Cryo-TEMPO by month |
| 02 | `NB02_common_mask_and_sectors.py` | Common mask (valid radar freeboard in all products), six longitude sectors, sector means |
| 03 | `NB03_snow_harmonisation.py` | Snow-harmonisation Cases A–D |
| 04 | `NB04_chapter3_figures.py` | Maps, difference maps, time series and scatter figures |
| 05 | `NB05_bootstrap_uncertainty.py` | Year-level bootstrap (1,000 realisations) for spread and spread reductions |
| 06 | `NB06_alternative_snow_reference_v2.py` | Alternative common snow references (Table A1) |
| 07 | `NB07_figures_with_uncertainty_v2.py` | Harmonisation figures with confidence intervals |
| 08 | `NB08_trend_sensitivity_tests.py` | Trend sensitivity (Tables 5 and 6; 2,000 bootstrap realisations) |
| 09 | `NB09_spread_vs_signal_v2.py` | Inter-product spread against seasonal-cycle amplitude and inter-annual range |
| 10 | `NB10_reconcile_spread_sources.py` | Distributed LEGOS sea-ice freeboard against Equation 1 |

Checks added for the second minor revision (November 2026):

| Script | Purpose |
|---|---|
| `C1_density_check.py` | Ratio of the distributed snow correction to snow thickness, by product and month |
| `C2_sens_CT300.py` | Reruns Cases A–D and the bootstrap with Cryo-TEMPO at 300 kg m⁻³ (Table A2) |
| `P1_offset_trend_pvalues.py` | Offset-trend p-values with a Student's t test (Table 6) |
| `P2_check_tables_2_3.py` | Recomputes Tables 2 and 3 from the unrounded multi-year means |

Terminal output of each script is in `output/`. `data/` mirrors the `processed/` directory used by the scripts.

---

## Reproducing the derived outputs

Create the conda environment, activate it, and run the steps in order:

```bash
conda env create -f environment.yml
conda activate ms1-freeboard
python scripts/NB00_inspect_raw_data.py
python scripts/NB01_regrid_to_cci_grid.py
python scripts/NB02_common_mask_and_sectors.py
python scripts/NB03_snow_harmonisation.py
python scripts/NB04_chapter3_figures.py
python scripts/NB05_bootstrap_uncertainty.py
python scripts/NB06_alternative_snow_reference_v2.py
python scripts/NB07_figures_with_uncertainty_v2.py
python scripts/NB08_trend_sensitivity_tests.py
python scripts/NB09_spread_vs_signal_v2.py
python scripts/NB10_reconcile_spread_sources.py
```

If you prefer `pip` over `conda`, an equivalent (Python ≥ 3.10) environment
can be constructed from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Each script sets its input and output directories near the top (for example BASE); change these to your own paths before running.

---

## Software environment

- Python ≥ 3.10
- Core scientific stack: `numpy`, `pandas`, `xarray`, `netCDF4`, `scipy`,
  `matplotlib`, `cartopy`, `pyproj`, `cmocean`
- Regridding: nearest-neighbour lookup with scipy.spatial.cKDTree;
- Bootstrap / statistics: `scipy.stats`

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
> Preprint: https://doi.org/10.5194/egusphere-2026-662

**Software / archived release**

> Liu, X., Tilling, R. L., Corney, S. P., Fraser, A. D., and Heil, P. (2026).
> *Python Scripts and Results for "Antarctic Sea-Ice Freeboard from
> Envisat and CryoSat-2: Attributing Inter-Product Spread to Snow and
> Radar-Retrieval Contributions"* (v1.1.0) [Software]. Zenodo.
> https://doi.org/10.5281/zenodo.19977864

---

## License

This repository is released under the [MIT License](LICENSE).

---

## Contact

Xinlong Liu  
Institute for Marine and Antarctic Studies, University of Tasmania  
xinlong.liu@utas.edu.au
