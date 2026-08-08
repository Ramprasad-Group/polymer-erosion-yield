# polymer-erosion-yield

Curated polymer atomic oxygen erosion yield dataset and a unified pipeline comparing a fine tuned LLM against Gaussian process regression on identical data splits.

## Overview

This repository contains a curated dataset of polymer atomic oxygen erosion yields compiled from low Earth orbit spaceflight experiments, together with the pipeline and results reported in the accompanying manuscript. The pipeline fine tunes and runs inference on OpenAI models, fits GPR with a Tanimoto + RBF kernel on Morgan fingerprints, optimizes both models, and evaluates both using the same restricted-group and random splitting strategies and metrics.

## Repository layout

```text
polymer-erosion-yield/
├── README.md
├── requirements.txt
├── pipeline.py
├── polymer_Ey_dataset_final.csv
├── random/
├── rg/
└── runs/
```

`random/` and `rg/` contain the split files used for the reported validation results. `runs/` contains saved predictions, metrics, figures, tuning results, and provenance information.

## Dataset

The target is the atomic oxygen erosion yield `e_y (A3/atom)`, the material volume removed per incident oxygen atom. The pipeline models its base 10 logarithm. The data was compiled from multiple NASA LEO missions and exposure studies.

| Column | Description |
| --- | --- |
| `psmiles` | Canonical PSMILES representation |
| `polymer name` | Polymer name |
| `coating name` | Coating applied to the material, if any |
| `mission name` | Source mission |
| `orientation` | Exposure orientation, ram / nadir / wake / zenith |
| `mission time (yr)` | Exposure duration in years |
| `solar exposure (esh)` | Equivalent Sun Hours |
| `ao fluence (atoms/cm2)` | Incident AO per unit area |
| `layers` | Number of thin film layers |
| `thickness (mm)` | Per layer thickness in millimeters |
| `e_y (A3/atom)` | Erosion yield, target, log10 transformed |

## Installation

```bash
pip install -r requirements.txt
```

## OpenAI API setup

LLM modes that submit fine tuning jobs or run inference require an OpenAI key. The pipeline checks `openai_api_key.txt` in the working directory and then the `OPENAI_API_KEY` environment setting. Do not commit API keys.

GPR modes and figure modes run without an OpenAI key.

## Pipeline modes

```bash
python pipeline.py <mode> --data_csv polymer_Ey_dataset_final.csv --rg_dir rg
```

| Mode | Does |
| --- | --- |
| `temp-tune` | Sweeps inference temperature over the restricted-group folds and writes `runs/llm_temp.json`. |
| `epoch-tune` | Fine tunes the epoch grid over the restricted-group folds and selects epochs by OME and log-R2. |
| `llm-cv` | Runs final LLM cross validation using the restricted-group and random splits. |
| `llm-report` | Scores completed LLM cross validation results. |
| `llm-prod` | Fine tunes one LLM on all 201 rows. |
| `llm-ablation` | Runs the layers and thickness feature ablation with matched control rows. |
| `gpr-opt` | Sweeps fingerprint bits, radius, and kernel and writes `runs/gpr_best.json`. |
| `gpr-cv` | Runs the selected GPR configuration using the restricted-group and random splits with predictive sigma. |
| `gpr-prod` | Fits one GPR on all 201 rows plus 10-fold generalization CV. |
| `gpr-ablation` | Runs the same feature ablation with GPR. |
| `validation-fig` | Creates the combined LLM and GPR parity figure from saved predictions. |
| `data-fig` | Creates the dataset description figure. |
| `tuning-fig` | Creates the epoch and temperature tuning figure. |
| `ablation-fig` | Creates the LLM ablation parity panels. |
| `gpr-ablation-fig` | Creates the GPR ablation parity panels. |
| `plot-only` | Redraws one saved predictions CSV. |

Figure and report modes make no API calls.

## Outputs

`runs/` contains the artifacts used in the paper, including metrics CSVs, per point prediction CSVs, parity figures, tuning results, and provenance information.

## Citation

If you use this dataset or pipeline, please cite the published work.

## Reproducibility and dataset correction

Three PSMILES entries, Kevlar, Nomex, and PFA, contained minor typos that were identified and corrected after the validation models were trained. The production models use the corrected dataset; to reproduce the reported validation results, do nothing, while to rerun the validation using the corrected dataset, delete both `random/` and `rg/` before running the pipeline.
