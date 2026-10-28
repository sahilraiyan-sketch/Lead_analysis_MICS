# Analysis code

Code for "Household and neighborhood components of survey indicators of child blood lead in Bangladesh". It regenerates the estimates, figures and results workbook from the Bangladesh MICS 2025 microdata. The repository contains no microdata.

## Data

1. Register at https://mics.unicef.org and request access, stating the research purpose.
2. After approval, open https://mics.unicef.org/surveys, select Bangladesh 2025 and download the SPSS archive.
3. Place `hh.sav`, `ch.sav` and `wm.sav` in one folder (`hl.sav`, `bh.sav` and `fs.sav` are not used).

## Setup

```
pip install -r requirements.txt
export MICS_DIR=/path/to/sav/folder
export OUT_DIR=/path/for/outputs
bash run_all.sh
```

`run_all.sh` runs the scripts in the order below. Scripts that write checkpoints (`run_atlas.py`, `boot.py`, `batch_checks.py`, `artifact_fe.py`) resume where they stopped, and `TIME_BUDGET` (seconds) can be set to stop them early. A full run takes about 30 minutes on one CPU.

## Files

| File | What it does | Main outputs | Used for |
|---|---|---|---|
| `config.py` | Reads `MICS_DIR` and `OUT_DIR` | | all scripts |
| `engine.py` | Survey-weighted linear and logistic estimation with Taylor-linearized variance over the full PSU frame; builds the analytic child cohort | | all scripts |
| `descriptives.py` | Cohort accounting, participant characteristics, completion of testing by division and wealth, weighted prevalences and descriptive numbers for the case study | `cohort_accounting.csv`, `cohort_characteristics.csv`, `completion.csv`, `prevalence_weighted.csv`, `case_descriptives.csv` | Tables S1, S1b, S4; Results 3.1 and 3.2 |
| `case_study.py` | Piped-water Models 1 to 4 and within-PSU conditional logit | `case_study_results.csv` | Table 1, Table S6 |
| `case_sensitivity.py` | Dhaka coverage estimate with leave-one-out coverage, the 3.5 ug/dL threshold, exclusion of the urban-industrial core and completion weighting | `case_sensitivity.csv` | Table S3 |
| `division_models.py` | District-adjusted models of piped-water use and coverage within each division | `division_models.csv` | Table S5 |
| `screen.py` | Defines the 81 candidate indicators and screens them on exposure variables only | `screen.json` | Table S2 |
| `hh_ind.py` | Computes household-level indicators and PSU means for all 62,980 households | `hhI.pkl`, `hh_psu_means.pkl` | input to `women_rep.py` |
| `build_ind.py` | Aligns indicators, PSU means, shrunken means and reliability with the child cohort | `H.pkl`, `C.pkl`, `CE.pkl`, `cohort.pkl`, `universe_stats.csv` | input to all later scripts |
| `add_collection_fields.py` | Adds phlebotomist, month, hour and laboratory-block fields to the cohort | `cohort.pkl` | artifact and batch checks |
| `run_atlas.py` | Total and within-between models for 63 indicators (pooled, Dhaka, elsewhere; linear and binary outcomes; shrunken-mean and no-wealth variants) | `atlas_raw.csv`, `status.csv` | Tables 2, S7, S8, S11 |
| `boot.py` | PSU-cluster bootstrap of the neighborhood share | `boot_partial.csv` | Table S11 |
| `classify.py` | Holm and Benjamini-Hochberg adjustment and the share-based classification | `final_pooled.csv` | Table 2, Supplementary methods S4 |
| `patterns.py` | Adjusted p-values across 63 indicators, descriptive patterns, SDs of household values, common-scale within components | `pooled_final_with_patterns.csv`, `sdH.csv` | Tables 2, S7; Figures 2, 3 |
| `women_rep.py` | Repeats the decomposition in pregnant women and compares it with children | `women_results.csv`, `women_vs_children.csv` | Table S9; Figure 2 |
| `women_leadmodule.py` | Total associations of lead-specific items in pregnant women | `women_leadmodule.csv` | Table S10 |
| `artifact_diag.py` | Residual intraclass correlations, within-PSU block tests, drift, leave-one-phlebotomist-out | `artifact_diag.csv`, `loo_phleb.csv` | Tables S12, S14 |
| `artifact_fe.py` | Decomposition with phlebotomist, month, hour and laboratory-block fixed effects | `artifact_fe.csv` | Table S13 |
| `batch_checks.py` | Laboratory-block random intercept; leave-one-block-out and leave-one-district-out refits | `batch_re.csv`, `batch_loo.csv` | Table S15 |
| `censor_sens.py` | Exclusion of results at the minimum value and unweighted Tobit models | `censor_sens.csv` | Table S16 |
| `figures.py` | Figures 2 and 3 on the common per-SD scale | `figures/` | Figures 2, 3 |
| `case_figures.py` | Figure 1 (prevalence by PSU coverage and household use), Figure S1 (cohort flow) and Figure S2 (model sequence) | `figures/` | Figure 1, Figures S1, S2 |
| `workbook.py` | Collects all estimates into one workbook | `Results_workbook.xlsx` | all tables |
| `run_all.sh` | Runs every script in order | | |

## Verification

Each script was run from the raw files in this packaged form. Estimates match the manuscript and supplement; the case-study Model 4 values (Table S13) are the ones this code produces. Indicator coding follows MICS and JMP conventions (Table S2 lists each rule) and should be checked against the UNICEF MICS syntax.
