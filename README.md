# polymer-erosion-yield

Curated polymer atomic oxygen erosion yield dataset and a unified pipeline comparing a fine-tuned LLM against Gaussian process regression on identical data splits.

## Overview

This repository contains a curated dataset of polymer atomic oxygen (AO) erosion yields compiled from low Earth orbit spaceflight experiments, together with the pipeline and results reported in the accompanying manuscript. The pipeline fine-tunes and runs inference on OpenAI GPT-4o, fits Gaussian process regression (GPR) with a Tanimoto kernel on Morgan fingerprints plus an RBF kernel on the numeric and categorical features, optimizes both models, and evaluates both on the same restricted-group and random splits with the same metrics.

## Repository layout

```text
polymer-erosion-yield/
├── README.md
├── .gitattributes
├── .gitignore
├── requirements.txt
├── pipeline.py
├── polymer_Ey_dataset_final.csv
├── rg/
└── runs/
```

`rg/` contains the five restricted-group train/test splits. The ten random folds are defined by `runs/random_split_membership.csv`. `runs/` contains the saved predictions, metrics, figures, tuning results and provenance information.

## Dataset

`polymer_Ey_dataset_final.csv` has 201 measurements covering 55 polymer chemistries. The prediction target is the base-10 logarithm of the atomic oxygen erosion yield `e_y (A3/atom)`, the material volume removed per incident oxygen atom.

| Column | Description |
| --- | --- |
| `polymer description` | Textual description of the polymer identity |
| `coating description` | Coating applied to the sample, if any |
| `psmiles` | Canonical PSMILES of the repeat unit |
| `layers` | Number of thin-film layers |
| `thickness (mils)` | Per-layer thickness in mils |
| `thickness (mm)` | Per-layer thickness in millimeters |
| `e_y (cm3/atom)` | Erosion yield in cm³/atom |
| `e_y (A3/atom)` | Erosion yield in Å³/atom |
| `log(e_y)` | Base-10 logarithm of the erosion yield in Å³/atom |
| `mission time (yr)` | Exposure duration in years |
| `solar exposure (esh)` | Solar exposure in equivalent sun hours |
| `ao fluence (atoms/cm2)` | Incident AO per unit area |
| `mission name` | Source mission |
| `orientation` | Exposure orientation: ram, nadir, wake or zenith |
| `paper` | Source of the erosion yield measurement |
| `PSMILES source` | Source used to determine the PSMILES representation |
| `PSMILES comment` | How the source informs the PSMILES representation |

## Installation

```bash
pip install -r requirements.txt
```

## OpenAI API setup

LLM modes that submit fine-tuning jobs or run inference require an OpenAI API key. The pipeline checks `openai_api_key.txt` in the working directory and then the `OPENAI_API_KEY` environment variable. Do not commit API keys.

Inference results are cached per row in `runs/`, so rerunning `temp-tune`, `epoch-tune`, `llm-cv` or `llm-report` against the saved `runs/` directory reproduces the reported LLM results without any API calls (`llm-cv` and `epoch-tune` still expect a key to be set). Fine-tuned model, job and file identifiers in `runs/` have been replaced with descriptive labels.

GPR modes and figure modes run without an OpenAI key.

## Pipeline modes

```bash
python pipeline.py <mode> --data_csv polymer_Ey_dataset_final.csv --rg_dir rg
```

| Mode | Does |
| --- | --- |
| `temp-tune` | Sweeps inference temperature over the restricted-group folds and writes `runs/llm_temp.json`. |
| `epoch-tune` | Compares training epochs over the restricted-group folds and writes `runs/llm_best.json`. |
| `llm-cv` | Fine-tunes and evaluates the LLM on the restricted-group (5) and random (10) folds. |
| `llm-report` | Scores completed LLM cross-validation folds. |
| `llm-prod` | Fine-tunes one LLM on all 201 rows. |
| `llm-ablation` | Runs the layers and thickness feature ablation on matched-control rows. |
| `gpr-opt` | Sweeps fingerprint bits, radius and kernel and writes `runs/gpr_best.json`. |
| `gpr-cv` | Runs the selected GPR configuration on the restricted-group and random folds. |
| `gpr-prod` | Fits one GPR on all 201 rows plus a 10-fold cross-validation estimate. |
| `validation-fig` | Creates the combined LLM and GPR parity figure from saved predictions. |
| `tuning-fig` | Creates the epoch and temperature tuning figure. |
| `ablation-fig` | Creates the LLM ablation parity figure. |
| `data-fig` | Creates the dataset description figure. |
| `plot-only` | Redraws one saved predictions CSV. |

Figure modes make no API calls and fit no models.

The reported results were produced in this order: `llm-cv` (initially scored at temperature 0.25), then `temp-tune` and `epoch-tune` (both reuse the `llm-cv` restricted-group models), then `llm-report` and `llm-ablation` at the selected temperature of 0.3; `gpr-opt`, then `gpr-cv`; then the figure modes.

## Outputs

`runs/` contains the artifacts used in the paper:

| Path | Contents |
| --- | --- |
| `runs/validation_figure.*` | LLM and GPR cross-validation parity plots |
| `runs/tuning_figure.*` | Epoch and inference temperature tuning |
| `runs/ablation_figure.*` | Layers and thickness ablation |
| `runs/dataset_figure.*` | Dataset description (`data-fig` output) |
| `runs/llm_cv/`, `runs/gpr_cv/` | Per-split and summary metrics, per-point predictions, parity plots and provenance |
| `runs/llm_temp_tune/`, `runs/llm_epoch_tune/`, `runs/temp_tune_results.csv`, `runs/epoch_tune_results.csv` | LLM tuning |
| `runs/llm_ablation/` | Ablation prompts, raw inference repeats and results |
| `runs/gpr_opt_results.csv` | GPR hyperparameter sweep |

Each fine-tuned unit directory holds its prompts (`train.jsonl`, `test.jsonl`), and every set of saved LLM predictions has its raw inference responses (`raw/rep*.csv`) alongside. The 5-epoch tuning arm reuses the `llm-cv` restricted-group models, so its responses are those in `runs/llm_cv/`. Refitting GPR with different library versions can shift its predictions slightly; the saved files in `runs/gpr_cv/` are the reported results.

## Citation

If you use this dataset or pipeline, please cite the published work.
