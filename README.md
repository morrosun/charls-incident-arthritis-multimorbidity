# Incident arthritis/rheumatism in CHARLS: metabolic multimorbidity and adiposity

Analysis code for a study of **genuinely incident** arthritis or rheumatism in middle-aged
and older Chinese adults, with replication in three further ageing cohorts.

The design treats the survey instrument as the exposure frame. In CHARLS wave 6 (2021–2023)
the diagnosis module asks about diagnoses *since the previous interview* only of respondents
recorded as free of the condition at that contact. The skip pattern therefore defines the
at-risk set and the new-diagnosis outcome before any analyst decision. The analysis contrasts
this incident outcome with the cumulative ever-diagnosed variable used in earlier waves, on a
common risk-ratio scale, and asks which baseline domains predict incident disease.

## What this repository contains

`analysis/` — 22 Python scripts, in four groups:

| Group | Scripts | Role |
|---|---|---|
| Cohort construction | `05_link.py`, `10_build.py`, `30_extend_w4.py` | link three release waves, resolve duplicate IDs to the latest release, apply the skip pattern, attach 2018 (wave-4) exposures |
| Primary analysis | `20_models.py`, `21_construct.py`, `31_analysis.py`, `35_final_models.py`, `38_cluster.py`, `40_mi.py`, `41_biomarker.py`, `42_rcs.py`, `43_cat.py`, `44_tables.py`, `80_tables_v4.py` | modified Poisson models, incident-vs-prevalent competing-risks comparison, 20-fold multiple imputation, lagged 2015 BMI and blood biomarkers, dose–response and restricted cubic splines, result tables |
| Figures | `36_figs.py`, `45_figs2.py`, `46_flow.py`, `65_fig_construct.py`, `99_figs_en.py`, `117_fig_external.py` | participant flow, forest plots, spline and construct-comparison figures, external-validation plots |
| External validation | `115_build_external.py`, `116_external_models.py` | build wave *t* → *t+1* incident transitions in HRS, ELSA and MHAS; refit the same exposure/covariate set and pool with DerSimonian–Laird |

The repository is a **code archive**, not a data package. Every table and figure is regenerated
from the scripts plus the source data; no numeric result is stored here.

## What this repository deliberately does not contain

- **Any respondent-level data.** CHARLS is distributed by Peking University's CHARLS team and
  the harmonised HRS/ELSA/MHAS files by the Gateway to Global Aging Data, each under
  registration and a data-use agreement. None may be redistributed. `*.dta`, `*.csv` and
  `data/` are excluded.
- **Manuscript, supplementary-material and report assembly code**, including the HTML/DOCX
  builders, the de-AI and polishing passes, and the reference-fetching and renumbering scripts.
  Those files embed the unpublished text and are not part of the analysis.
- **Derived result tables and log files** (`out/`, `outputs/`), which are created at run time
  and git-ignored.

## Data and environment variables

Obtain CHARLS wave 6 (2021/2022/2023) and wave-3 (2015) files from the CHARLS team, and the
harmonised HRS, ELSA and MHAS files from <https://g2aging.org/>, then place them on your machine
and point the following variables at them. Each has a portable default under `<repo>/data`.

| Variable | Default | Meaning |
|---|---|---|
| `CHARLS_RAW` | `<repo>/data/raw/CHARLS` | directory of the three wave-6 annual packages (each containing `Health_Status_and_Functioning.dta` and `Demographics.dta`) |
| `CHARLS_WORK` | `<repo>/data/working/charls.dta` | CHARLS harmonised working file (`Working_data/charls.dta`) |
| `HRS_FILE` | `<repo>/data/working/hrs.dta` | HRS harmonised working file |
| `ELSA_FILE` | `<repo>/data/working/elsa.dta` | ELSA harmonised working file |
| `MHAS_FILE` | `<repo>/data/working/mhas.dta` | MHAS harmonised working file |
| `W6_OUT` | `<repo>/out` | directory for all generated results |

No credentials are required: all inputs are local files. The scripts read no network services.

## Reproducing the analysis

Python 3.11+ with `pandas`, `numpy`, `pyreadstat`, `scipy`, `scikit-learn`, `statsmodels`,
`matplotlib`. Run from the repository root, roughly in this order (later scripts consume the
CSV/TXT files written by earlier ones into `$W6_OUT`; a few early scripts are alternative
stages and are retained for provenance):

```bash
python analysis/10_build.py           # build the cohort from the three release waves
python analysis/30_extend_w4.py       # attach 2018 exposures
python analysis/38_cluster.py         # primary model with community-clustered SE
python analysis/40_mi.py              # 20-fold multiple imputation, Rubin combination
python analysis/41_biomarker.py       # lagged 2015 blood markers
python analysis/42_rcs.py             # restricted cubic splines
python analysis/43_cat.py             # categorical dose–response
python analysis/21_construct.py       # incident vs prevalent construct comparison
python analysis/115_build_external.py # build external transitions (HRS/ELSA/MHAS + CHARLS)
python analysis/116_external_models.py# refit and pool external estimates
python analysis/46_flow.py            # participant flow figure
python analysis/99_figs_en.py         # main forest/spline figures
python analysis/117_fig_external.py   # external-validation forest figure
```

Figure scripts consume result files produced by the model scripts and write PNGs into `$W6_OUT`.

## Citation and licence

This repository is archived on Zenodo; see the DOI badge and `.zenodo.json`. The archived
release is frozen at its tag; this README is a living document. Code is released under the
MIT licence (see `LICENSE`).
