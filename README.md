# polymer-erosion-yield

Curated polymer atomic-oxygen erosion-yield dataset and pipeline comparing a fine-tuned LLM with Gaussian process regression on identical validation splits.

## Overview

This repository contains a curated dataset of polymer atomic oxygen (AO) erosion yields compiled from low Earth orbit (LEO) spaceflight experiments, together with the pipeline and saved results reported in the accompanying manuscript. The paper validation comparison uses the same restricted-group and random splits for the fine-tuned LLM and GPR, with identical metrics. The pipeline also retains the target-stratified variable-split utilities and artifacts used during development.

## Repository layout

```text
polymer-erosion-yield/
├── README.md
├── requirements.txt
├── pipeline.py                          # unified pipeline
├── polymer_Ey_dataset_final.csv         # corrected canonical dataset (201 rows)
├── random/                              # exact historical paper random-split CSVs
├── rg/                                  # exact historical paper restricted-group CSVs
└── runs/
    ├── restricted-group_split_membership.csv
    ├── random_split_membership.csv
    ├── variable_split_membership.csv
    └── ...                              # saved predictions, metrics, figures, tuning and provenance
```

## Dataset

The target is the atomic-oxygen erosion yield `e_y (A3/atom)`, the material volume removed per incident oxygen atom. The pipeline models its base-10 logarithm. The data were compiled from multiple NASA LEO missions and exposure studies.

| Column | Description |
| --- | --- |
| `psmiles` | Canonical PSMILES representation |
| `polymer name` | Polymer name |
| `coating name` | Coating applied to the material, if any |
| `mission name` | Source mission |
| `orientation` | Exposure orientation (ram / nadir / wake / zenith) |
| `mission time (yr)` | Exposure duration in years |
| `solar exposure (esh)` | Equivalent Sun Hours |
| `ao fluence (atoms/cm2)` | Incident AO per unit area |
| `layers` | Number of thin-film layers |
| `thickness (mm)` | Per-layer thickness in millimeters |
| `e_y (A3/atom)` | Erosion yield (target; log10 transformed for modeling) |

## Split definitions

### Random split

The random result is genuine 10-fold cross-validation using `KFold(n_splits=10, shuffle=True, random_state=42)`. Every one of the 201 dataset rows is assigned to exactly one test fold, giving 201 pooled out-of-fold predictions. The exact historical paper train/test CSVs are bundled in `random/split_01..10`, and the row-to-fold assignment is also frozen in `runs/random_split_membership.csv`.

### Restricted-group split

The restricted-group validation contains 29 singleton chemistries divided across five held-out folds of 6/6/6/6/5. The exact paper fold membership is frozen by master-dataset row index in `runs/restricted-group_split_membership.csv`.

The bundled `rg/split_01..05` train/test CSVs are the exact historical restricted-group inputs used for the paper. Together, top-level `random/` and `rg/` are the executable paper-era CV inputs. They must be present or absent as a pair: with both present, retraining uses the exact paper inputs; with both deliberately deleted, random and restricted-group are rebuilt from the corrected 201-row master dataset using their frozen row-index memberships. A one-folder state is rejected to prevent an accidental mixture of historical and corrected inputs.

For independent restricted-group regeneration from the master CSV, singleton chemistries are taken in master-dataset row order, shuffled with `np.random.default_rng(42)`, and divided with `np.array_split(..., 5)`. Preserving master-row order before the shuffle reproduces the saved RG fold assignment exactly.

## Installation

```bash
pip install -r requirements.txt
```

## Quick start: use the saved results

```bash
python pipeline.py gpr-cv --data_csv polymer_Ey_dataset_final.csv
python pipeline.py validation-fig --data_csv polymer_Ey_dataset_final.csv --rg_dir rg
```

The shipped `runs/llm_cv` and `runs/gpr_cv` directories are frozen paper-result caches. Ordinary CV commands reuse/skip the existing results and do not retrain. Exact paper reproduction and corrected-data rerun instructions are given in the final reproducibility section below.

## OpenAI API setup

LLM modes that submit fine-tuning jobs or perform uncached inference require an OpenAI key, resolved in this order:

1. `openai_api_key.txt` in the working directory (one line; this filename is ignored by Git), or
2. the `OPENAI_API_KEY` environment variable.

No API key is stored in this repository. GPR and figure-only modes do not require an OpenAI key.

## Pipeline modes

```bash
python pipeline.py <mode> --data_csv polymer_Ey_dataset_final.csv --rg_dir rg
```

| Mode | Does |
| --- | --- |
| `temp-tune` | Sweeps inference temperature over RG 5-fold. |
| `epoch-tune` | Fine-tunes the epoch grid over RG 5-fold. |
| `llm-cv` | LLM CV workflow using the selected configuration. |
| `llm-report` | Scores available `llm-cv` models/predictions; uncached completed units can require inference. |
| `llm-prod` | Fine-tunes one LLM on all 201 rows. |
| `llm-ablation` | Runs the layers/thickness feature ablation with matched-control rows. |
| `gpr-opt` | Sweeps fingerprint bits, radius and kernel. |
| `gpr-cv` | GPR CV; existing saved strategy predictions are immutable unless deliberately deleted first. |
| `gpr-prod` | Fits one GPR on all 201 rows plus 10-fold generalization CV. |
| `gpr-ablation` | Runs the same feature ablation with GPR. |
| `validation-fig` | Creates the fixed 2x2 LLM/GPR validation figure for restricted-group and random from saved predictions. |
| `data-fig` | Creates the dataset-description figure. |
| `descriptor-fig` | Creates pairwise numerical-descriptor scatterplots. |
| `tuning-fig` | Creates the epoch/temperature tuning figure from saved artifacts. |
| `ablation-fig` | Creates the LLM ablation parity panels from saved artifacts. |
| `gpr-ablation-fig` | Creates the GPR ablation parity panels from saved artifacts. |
| `plot-only` | Redraws one saved `*_predictions.csv`. |
| `rebuild-bands` | Rebuilds cached LLM uncertainty bands from saved repeat files without changing predictions. |

## Citation

If you use this dataset or pipeline, please cite the accompanying published work.

## Reproducibility, saved paper results, and dataset revision

The paper cross-validation prediction artifacts in `runs/` are intentionally preserved exactly as generated. The corrected canonical dataset was finalized afterward. Relative to the paper-era inputs, it corrects the PSMILES for the Kevlar entries, Teflon PFA 200 CLP and Nomex T-410, corrects the Nomex T-410 material name, and includes the finalized coating cleanup. The pipeline also repairs the known indium-tin-oxide-coated silver-Teflon coating field at load time.

The repository preserves both the exact paper-era validation inputs and the corrected canonical dataset. The two symmetric top-level split folders are the deliberate data-source switch:

- `random/` **and** `rg/` both present -> exact paper-era random and restricted-group inputs.
- `random/` **and** `rg/` both deleted -> both splits are rebuilt from the corrected `polymer_Ey_dataset_final.csv` using the frozen row-index memberships in `runs/`.
- only one present -> the pipeline stops rather than mixing historical and corrected inputs.

The shipped `runs/llm_cv` and `runs/gpr_cv` directories are frozen paper-result caches. Deleting a whole model CV directory is the explicit retraining opt-in. Normal commands leave those results unchanged.

To retrain the paper validation splits exactly, leave `random/` and `rg/` in place and delete only the model cache you intend to rebuild:

```bash
# exact paper-era GPR validation retrain
rm -rf runs/gpr_cv
python pipeline.py gpr-cv --split restricted-group
python pipeline.py gpr-cv --split random

# exact paper-era LLM validation retrain
rm -rf runs/llm_cv
python pipeline.py llm-cv --split restricted-group
python pipeline.py llm-cv --split random
```

To switch both validation splits to the corrected dataset, deliberately remove both paper-input folders, then remove whichever model cache you want to retrain:

```bash
# one explicit data-source switch for BOTH validation splits
rm -rf random rg

# then choose the model(s) to retrain
rm -rf runs/gpr_cv
python pipeline.py gpr-cv --split restricted-group
python pipeline.py gpr-cv --split random

rm -rf runs/llm_cv
python pipeline.py llm-cv --split restricted-group
python pipeline.py llm-cv --split random
```

Do **not** delete `runs/random_split_membership.csv` or `runs/restricted-group_split_membership.csv` for a corrected-data rerun. They store only the intended fold membership; once `random/` and `rg/` are absent, the actual row contents are read from the corrected master dataset.

For LLM CV, the pipeline refuses a partial rebuild if strategy-level results are missing while old per-fold/model state remains, preventing accidental reuse of historical fine-tuned model IDs. Saved fine-tuned model IDs belong to the original OpenAI organization; rebuilding under another account requires new fine-tuning and is separate from reading the saved paper artifacts.

Figure modes use saved prediction files; they do not retrain either model.
