# Analysis code

Code for "Household and neighborhood components of survey indicators of child blood lead in Bangladesh". It regenerates every estimate, table, figure and the results workbook from the Bangladesh MICS 2025 microdata. The folder contains no microdata.

## 1. Get the data

1. Register at https://mics.unicef.org and request access, stating the research purpose.
2. After approval, open https://mics.unicef.org/surveys, select Bangladesh 2025 and download the SPSS archive.
3. Put `hh.sav`, `ch.sav` and `wm.sav` in one folder. The other files in the archive are not used.

## 2. Install

Python 3.10 or later, then:

```
pip install -r requirements.txt
```

## 3. Run

One command runs everything in order (about 30 minutes on one CPU). It works the same on Windows, macOS and Linux, including the RStudio terminal.

Windows example:

```
python run_all.py --data "C:\mics2025" --out "C:\mics_results"
```

macOS or Linux example:

```
python run_all.py --data /home/me/mics2025 --out /home/me/mics_results
```

Options:

- `--from boot.py` starts at a given script (use it to resume after an interruption).
- `--only workbook.py` runs a single script.

Scripts that save checkpoints (`run_atlas.py`, `boot.py`, `batch_checks.py`, `artifact_fe.py`) also resume where they stopped when run again.

To run one script by itself, set the two folders first and then call Python:

```
PowerShell:  $env:MICS_DIR="C:\mics2025"; $env:OUT_DIR="C:\mics_results"; python screen.py
cmd:         set MICS_DIR=C:\mics2025 & set OUT_DIR=C:\mics_results & python screen.py
bash:        MICS_DIR=/path/data OUT_DIR=/path/out python screen.py
```

## 4. What each file does

Shared code
- `config.py`: reads the data and output folders.
- `engine.py`: survey-weighted linear and logistic estimation with Taylor-linearized variance over the full PSU frame; builds the analytic child cohort.
- `run_all.py`: runs the scripts below in order.

Piped-water case study (main Table 1, Figure 1)
- `descriptives.py`: cohort accounting, participant characteristics, completion of testing, weighted prevalences. Output: `cohort_accounting.csv`, `cohort_characteristics.csv`, `completion.csv`, `prevalence_weighted.csv`, `case_descriptives.csv`. Used for Tables S1, S1b, S4 and Results 3.1 and 3.2.
- `case_study.py`: Models 1 to 4 and the within-PSU conditional logit. Output: `case_study_results.csv`. Used for Table 1 and Table S6.
- `case_sensitivity.py`: leave-one-out coverage, the 3.5 ug/dL threshold, exclusion of the urban-industrial core, completion weighting. Output: `case_sensitivity.csv`. Used for Table S3.
- `division_models.py`: district-adjusted models within each division. Output: `division_models.csv`. Used for Table S5.
- `case_figures.py`: Figure 1, Figure S1 and Figure S2. Output: `figures/`.

Indicator decomposition (main Table 2, Figures 2 and 3)
- `screen.py`: defines the 81 candidate indicators and screens them on exposure variables only. Output: `screen.json`. Used for Table S2.
- `hh_ind.py`: household-level indicators and PSU means for all 62,980 households. Output: `hhI.pkl`, `hh_psu_means.pkl`.
- `build_ind.py`: aligns indicators, PSU means, shrunken means and reliability with the child cohort. Output: `H.pkl`, `C.pkl`, `CE.pkl`, `cohort.pkl`, `universe_stats.csv`.
- `add_collection_fields.py`: adds phlebotomist, month, hour and laboratory-block fields to the cohort.
- `run_atlas.py`: total and within-between models for 63 indicators (pooled, Dhaka, elsewhere; linear and binary outcomes; shrunken-mean and no-wealth variants). Output: `atlas_raw.csv`, `status.csv`. Used for Tables 2, S7, S8 and S11.
- `boot.py`: PSU-cluster bootstrap of the neighborhood share. Output: `boot_partial.csv`. Used for Table S11.
- `classify.py`: Holm and Benjamini-Hochberg adjustment and the share-based classification. Output: `final_pooled.csv`. Used for Table 2 and Supplementary methods S4.
- `patterns.py`: adjusted p-values across 63 indicators, descriptive patterns, SDs of household values, common-scale within components. Output: `pooled_final_with_patterns.csv`, `sdH.csv`. Used for Tables 2 and S7.
- `figures.py`: Figures 2 and 3. Output: `figures/`.

Replication in pregnant women
- `women_rep.py`: repeats the decomposition in pregnant women and compares it with children. Output: `women_results.csv`, `women_vs_children.csv`. Used for Table S9 and Figure 2.
- `women_leadmodule.py`: lead-specific items in pregnant women. Output: `women_leadmodule.csv`. Used for Table S10.

Artifact, batch and detection-limit checks
- `artifact_diag.py`: residual intraclass correlations, within-PSU block tests, drift, leave-one-phlebotomist-out. Output: `artifact_diag.csv`, `loo_phleb.csv`. Used for Tables S12 and S14.
- `artifact_fe.py`: decomposition with phlebotomist, month, hour and laboratory-block fixed effects. Output: `artifact_fe.csv`. Used for Table S13.
- `batch_checks.py`: laboratory-block random intercept; leave-one-block-out and leave-one-district-out refits. Output: `batch_re.csv`, `batch_loo.csv`. Used for Table S15.
- `censor_sens.py`: exclusion of results at the minimum value and unweighted Tobit models. Output: `censor_sens.csv`. Used for Table S16.

Collecting results
- `workbook.py`: gathers all estimates into `Results_workbook.xlsx`.

## 5. Notes

- Table numbers refer to the supplement, numbered in order of first citation in the manuscript.
- The case-study Model 4 estimates in Table S6 are the ones this code produces.
- Indicator coding follows MICS and JMP conventions (Table S2 lists each rule) and should be checked against the UNICEF MICS syntax.
- Each script was run from the raw files in this packaged form.
