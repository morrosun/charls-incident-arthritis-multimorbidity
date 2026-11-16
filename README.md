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

`analysis/` — 24 Python scripts, in four groups:

| Group | Scripts | Role |
|---|---|---|
| Cohort construction | `05_link.py`, `10_build.py`, `30_extend_w4.py` | link three release waves, resolve duplicate IDs to the latest release, apply the skip pattern, attach 2018 (wave-4) exposures |
| Primary analysis | `20_models.py`, `31_analysis.py`, `35_final_models.py`, `38_cluster.py`, `40_mi.py`, `41_biomarker.py`, `42_rcs.py`, `43_cat.py`, `44_tables.py`, `80_tables_v4.py` | modified Poisson models, 20 multiple imputations, lagged 2015 BMI and blood biomarkers, dose–response and restricted cubic splines, result tables |
| Construct comparison | `21_construct.py`, `22_construct_precov.py`, `23_construct_competing.py` | the same associations under four outcome definitions, and a formal incident-versus-prevalent comparison |
| Figures | `36_figs.py`, `45_figs2.py`, `46_flow.py`, `65_fig_construct.py`, `99_figs_en.py`, `117_fig_external.py` | participant flow, forest plots, spline and construct-comparison figures, external-validation plots |
| External validation | `115_build_external.py`, `116_external_models.py` | build wave *t* → *t+1* incident transitions in HRS, ELSA and MHAS; refit the same exposure/covariate set and pool with DerSimonian–Laird |

Four scripts are retained for provenance but are **not** part of the final pipeline, and each
says so in its header: `05_link.py` (an early exploration whose de-duplication rule was
superseded), `20_models.py` and `31_analysis.py` (earlier model specifications, kept to show
the sequence of adjustment sets), and `21_construct.py` (offset and HC0 variance, superseded by
`22_construct_precov.py`). They write no input for any downstream step.

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
Every script routes its outputs through `W6_OUT`, so all results land in one directory.

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

Python 3.10+ with the packages pinned in `requirements.txt` (`pandas`, `numpy`, `pyreadstat`,
`scipy`, `scikit-learn`, `statsmodels`, `matplotlib`, `patsy`). Run from the repository root.
Later scripts consume the CSV/JSON files written by earlier ones into `$W6_OUT`:

```bash
python analysis/10_build.py            # build the cohort from the three release waves
python analysis/30_extend_w4.py        # attach 2018 exposures; writes analytic.csv
python analysis/38_cluster.py          # add communityID; primary model, clustered SE
python analysis/40_mi.py               # 20 multiple imputations, Rubin combination
python analysis/22_construct_precov.py # four outcome definitions, pre-outcome covariates
python analysis/23_construct_competing.py # incident vs prevalent, interaction model
python analysis/41_biomarker.py        # lagged 2015 blood markers and BMI
python analysis/42_rcs.py              # restricted cubic splines
python analysis/43_cat.py              # categorical dose-response
python analysis/115_build_external.py  # build external transitions (HRS/ELSA/MHAS)
python analysis/116_external_models.py # refit and pool external estimates
python analysis/99_figs_en.py          # Figures 1-5
python analysis/117_fig_external.py    # Figure 6
```

`38_cluster.py` reads `analytic.csv`, adds `communityID` from the wave-6 `Demographics.dta`
files and writes the file back; it is idempotent, and running it more than once is harmless.
Figure scripts consume result files produced by the model scripts and write PNGs into `$W6_OUT`.

`10_build.py` also writes `flow_counts.json`, which holds every count shown in the participant
flow diagram (records, duplicates, at-risk set, exclusions, events, person-years, interview
batches). `99_figs_en.py` and `46_flow.py` read that file, so the diagram cannot drift away from
the data. No sample size is typed into a plotting script.

Result CSVs carry the study's internal exposure labels (Chinese) as the `exposure` column;
`80_tables_v4.py` holds the one-to-one mapping to the terms used in the manuscript.

## Citation and licence

This repository is archived on Zenodo; see the DOI badge and `.zenodo.json`. The archived
release is frozen at its tag; this README is a living document. Code is released under the
MIT licence (see `LICENSE`).
