# polymer-erosion-yield

Curated polymer atomic-oxygen erosion-yield dataset and pipeline comparing a fine-tuned LLM with Gaussian process regression on identical data splits.

## Overview

This repository contains a curated dataset of polymer atomic oxygen (AO) erosion yields compiled from low Earth orbit (LEO) spaceflight experiments, together with the pipeline and results reported in the accompanying manuscript. The pipeline fine-tunes and runs inference on OpenAI models, fits GPR with a Tanimoto + RBF kernel on Morgan fingerprints, optimizes both models and evaluates them using the same three split strategies (restricted-group and random) and metrics.

## Repository layout

```text
polymer-erosion-yield/
├── README.md
├── requirements.txt
├── pipeline.py                      # unified pipeline (all modes)
├── polymer_Ey_dataset_final.csv     # dataset (201 rows)
├── rg/                              # restricted-group split_01..05 train/test CSVs
└── runs/                            # predictions, figures, tuning results, provenance
```

## Dataset

The target is the atomic-oxygen erosion yield `e_y (A3/atom)`, the material volume removed per incident oxygen atom. The pipeline models its base-10 logarithm. The data was compiled from multiple NASA LEO missions and exposure studies.

| Column                   | Description                                        |
| ------------------------ | -------------------------------------------------- |
| `psmiles`                | Canonical PSMILES representation                   |
| `polymer name`           | Polymer name                                       |
| `coating name`           | Coating applied to the material, if any            |
| `mission name`           | Source mission                                     |
| `orientation`            | Exposure orientation (ram / nadir / wake / zenith) |
| `mission time (yr)`      | Exposure duration in years                         |
| `solar exposure (esh)`   | Equivalent Sun Hours                               |
| `ao fluence (atoms/cm2)` | Incident AO per unit area                          |
| `layers`                 | Number of thin-film layers                         |
| `thickness (mm)`         | Per-layer thickness in millimeters                 |
| `e_y (A3/atom)`          | Erosion yield (**target**, log10 transformed)      |

## Installation

```bash
pip install -r requirements.txt
```

### Quick start (no API key)

```bash
python pipeline.py gpr-cv --data_csv polymer_Ey_dataset_final.csv --rg_dir rg
python pipeline.py validation-fig --data_csv polymer_Ey_dataset_final.csv --rg_dir rg
```

GPR fits in minutes on a laptop. Figure modes redraw the saved predictions in `runs/`. LLM modes require an API key and incur fine-tuning costs.

## OpenAI API setup

LLM modes that submit fine-tuning jobs or run inference require an OpenAI key, resolved in this order:

1. `openai_api_key.txt` in the working directory (one line; do not commit this file), or
2. the `OPENAI_API_KEY` environment variable.

No key is stored in the code. GPR and figure modes run without a key.

## Pipeline modes

```bash
python pipeline.py <mode> --data_csv polymer_Ey_dataset_final.csv --rg_dir rg
```

| Mode               | Does                                                                        |
| ------------------ | --------------------------------------------------------------------------- |
| `temp-tune`        | Sweeps inference temperature over RG 5-fold; writes `runs/llm_temp.json`.   |
| `epoch-tune`       | Fine-tunes the epoch grid over RG 5-fold; selects epochs by OME/log-R2.      |
| `llm-cv`           | Runs final LLM CV using the selected configuration on all three split strategies. |
| `llm-report`       | Performs read-only scoring of completed `llm-cv` models.                    |
| `llm-prod`         | Fine-tunes one LLM on all 201 rows.                                         |
| `llm-ablation`     | Runs the layers/thickness feature ablation with matched-control rows.       |
| `gpr-opt`          | Sweeps fingerprint bits, radius and kernel; writes `runs/gpr_best.json`.    |
| `gpr-cv`           | Runs the selected GPR configuration on all three split strategies, with predictive sigma. |
| `gpr-prod`         | Fits one GPR on all 201 rows plus 10-fold generalization CV.                |
| `gpr-ablation`     | Runs the same feature ablation with GPR.                                    |
| `validation-fig`   | Creates the combined LLM/GPR parity figure across splits from saved predictions. |
| `data-fig`         | Creates the dataset-description figure.                                    |
| `tuning-fig`       | Creates the epoch and temperature tuning figure.                           |
| `ablation-fig`     | Creates the LLM ablation parity panels.                                     |
| `gpr-ablation-fig` | Creates the GPR ablation parity panels.                                     |
| `plot-only`        | Redraws one saved `*_predictions.csv`.                                      |

Figure and report modes make no API calls. Splits are frozen to membership files under `runs/` keyed by dataset row index, so reruns use identical folds.

## Outputs

`runs/` contains the artifacts used in the paper: per-split and pooled metrics CSVs (OME and log-R2 with 10000-resample bootstrap 95% CIs), per-point prediction CSVs, parity figures (SVG / EPS / PDF), tuning results and a provenance JSON for each result tying it to the dataset hash, split membership and model configuration.

Figure and report modes use the saved predictions in `runs/`. Refitting modes (`gpr-*`) validate the saved results against the current dataset and recompute them when they do not match, so a fresh run regenerates all GPR results from the corrected dataset. The `"FAILED validation -- recomputing"` message on a fresh clone indicates that this validation is working as intended.

LLM modes reuse the fine-tuned models recorded in `runs/`, which are private to our OpenAI organization. To rebuild the LLM results from scratch under your own key, pass a fresh `--out_dir`.

## Citation

If you use this dataset or pipeline, please cite the published work.

## Note

The results in `runs/` were generated with a prior dataset revision that differed only in the PSMILES of three polymers (Teflon PFA 200 CLP, Kevlar 29 and Nomex T-410). This repository ships the corrected dataset, and because both models saw identical inputs, the comparison is unaffected. The final production model was trained on the fully corrected dataset; only the cross-validation results predate the correction.
