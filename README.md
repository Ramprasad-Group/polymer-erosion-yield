# polymer-erosion-yield

Curated polymer atomic-oxygen erosion-yield dataset and a unified pipeline comparing a fine-tuned LLM against Gaussian process regression on identical data splits.

## Overview

This repository contains a curated dataset of polymer atomic oxygen (AO) erosion yields compiled from low Earth orbit (LEO) spaceflight experiments, together with the pipeline and results reported in the accompanying manuscript. The pipeline fine-tunes and runs inference on OpenAI models, fits GPR with a Tanimoto + RBF kernel on Morgan fingerprints, optimizes both, and evaluates both on the same three split strategies (restricted-group and random) with the same metrics, so the comparison comes from one reproducible file.

## Repository layout

```
polymer-erosion-yield/
├── README.md
├── requirements.txt
├── pipeline.py                      # unified pipeline (all modes)
├── polymer_Ey_dataset_final.csv     # dataset (201 rows)
├── rg/                              # restricted-group split_01..05 train/test CSVs
└── runs/                            # predictions, figures, tuning results, provenance
```

## Dataset

The target is the atomic-oxygen erosion yield `e_y (A3/atom)`, the material volume removed per incident oxygen atom; the pipeline models its base-10 logarithm. Data is compiled from multiple NASA LEO missions and exposure studies.

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

```
pip install -r requirements.txt
```

### Quick start (no API key)

```
python pipeline.py gpr-cv --data_csv polymer_Ey_dataset_final.csv --rg_dir rg
python pipeline.py validation-fig --data_csv polymer_Ey_dataset_final.csv --rg_dir rg
```

GPR fits in minutes on a laptop; figure modes redraw from the saved predictions in `runs/`. LLM modes need an API key and spend real money on fine-tunes.

## OpenAI API setup

LLM modes that submit fine-tune jobs or run inference need an OpenAI key, resolved in this order:

1. `openai_api_key.txt` in the working directory (one line; do not commit this file), or
2. the `OPENAI_API_KEY` environment variable.

No key is stored in the code. GPR modes and all figure modes run without a key.

## Pipeline modes

```
python pipeline.py <mode> --data_csv polymer_Ey_dataset_final.csv --rg_dir rg
```

| Mode               | Does                                                                        |
| ------------------ | --------------------------------------------------------------------------- |
| `temp-tune`        | Sweep inference temperature over RG 5-fold; writes `runs/llm_temp.json`.    |
| `epoch-tune`       | Fine-tune the epoch grid over RG 5-fold; picks best epochs by OME/log-R2.   |
| `llm-cv`           | Final LLM CV: winning config on all three split strategies.                 |
| `llm-report`       | Read-only scoring of whatever llm-cv models are finished.                   |
| `llm-prod`         | One LLM fine-tuned on all 201 rows.                                         |
| `llm-ablation`     | Layers/thickness feature ablation with matched-control rows.                |
| `gpr-opt`          | Sweep fingerprint bits, radius and kernel; writes `runs/gpr_best.json`.     |
| `gpr-cv`           | Winning GPR config on all three split strategies, with predictive sigma.    |
| `gpr-prod`         | One GPR on all 201 rows plus 10-fold generalization CV.                     |
| `gpr-ablation`     | The same feature ablation with GPR.                                         |
| `validation-fig`   | Combined LLM/GPR parity figure across splits, from saved predictions.       |
| `data-fig`         | Dataset-description figure.                                                 |
| `tuning-fig`       | Epoch and temperature tuning figure.                                        |
| `ablation-fig`     | LLM ablation parity panels.                                                 |
| `gpr-ablation-fig` | GPR ablation parity panels.                                                 |
| `plot-only`        | Redraw one saved `*_predictions.csv`.                                       |

Figure and report modes make no API calls and are safe to rerun. Splits are frozen to membership files under `runs/` keyed by dataset row index, so every rerun uses identical folds.

## Outputs

`runs/` contains the paper's exact artifacts: per-split and pooled metrics CSVs (OME and log-R2 with 10000-resample bootstrap 95% CIs), per-point prediction CSVs, parity figures (SVG / EPS / PDF), tuning results and a provenance JSON for each result tying it to the dataset hash, split membership and model configuration.

Figure and report modes always use the saved predictions in `runs/` as published. Refitting modes (`gpr-*`) validate saved results against the current dataset and recompute from it when they do not match, so a fresh run regenerates all GPR results from the corrected dataset; the "FAILED validation -- recomputing" message on a fresh clone is this working as intended. LLM modes reuse the fine-tuned models recorded in `runs/`, which are private to our OpenAI organization; to rebuild the LLM results from scratch under your own key, pass a fresh `--out_dir`.

## Citation

If you use this dataset or pipeline, please cite the published work.

## Note

The results in `runs/` were generated with a prior dataset revision that differed only in the PSMILES of three polymers (Teflon PFA 200 CLP, Kevlar 29 and Nomex T-410); this repository ships the corrected dataset, and since both models saw identical inputs the comparison is unaffected. The final production model was trained on the fully corrected dataset; only the cross-validation results predate the correction.
