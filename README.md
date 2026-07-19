# polymer-erosion-yield

Curated polymer atomic-oxygen erosion-yield dataset and a unified pipeline for fine-tuning and evaluating an LLM-based erosion-yield predictor against traditional ML.

## Overview

This repository contains a curated dataset of polymer atomic oxygen (AO) erosion yields compiled from low Earth orbit (LEO) spaceflight experiments, together with the pipeline used and results reported in the accompanying manuscript. The pipeline prepares restricted group and random data splits, fine-tunes and runs inference on OpenAI models, autotunes the LLM configuration, generates the parity plots, and benchmarks the LLM against an optimized GPR

The pipeline is distributed as a single script, `scripts/leo_pipeline.py`, driven by a `mode` argument. Keeping the split rule, canonicalizer, metrics, and both prediction paths in one source file is deliberate: it guarantees the LLM and ML benchmarks always share the same splits and evaluation, so the paper's numbers come from one reproducible file rather than a chain of scripts that can drift out of sync.

## Repository layout

```
polymer-erosion-yield/
├── README.md
├── data/
│   └── polymer_Ey_dataset_final.csv     # dataset
├── scripts/
│   └── leo_pipeline.py                   # unified pipeline (all modes)
├── splits/                               # the paper's exact CSV + JSONL splits
└── results/                              # the paper's predictions, figures, ML benchmark
```

## Dataset

The primary target is the log atomic-oxygen erosion yield, `log(e_y)` — the log of the material volume removed per incident oxygen atom. Data is compiled from multiple LEO missions and exposure studies (MISSE, EOIM, LDEF, STS, and related).

The pipeline reads the following columns:

| Column                     | Description                                       |
| -------------------------- | ------------------------------------------------- |
| `psmiles`                  | Polymer repeat-unit pSMILES representation        |
| `polymer name`             | Polymer name                                      |
| `coating name`             | Coating applied to the material, if any           |
| `mission name`             | Source mission (MISSE eligibility keys off this)  |
| `orientation`              | Exposure orientation (ram / nadir / wake / zenith)|
| `mission time (yr)`        | Exposure duration in years                        |
| `solar exposure (esh)`     | Equivalent Sun Hours                              |
| `ao fluence (atoms/cm2)`   | Atomic oxygen fluence                             |
| `layers`                   | Number of layers                                  |
| `thickness (mm)`           | Per-layer thickness in millimeters                |
| `log(e_y)`                 | Log-transformed erosion yield (**target**)        |

## Installation

The core LLM pipeline (`prep` / `train` / `retrieve` / `infer` / `figures` / `autotune`) needs:

```
pip install openai pandas numpy matplotlib
pip install git+https://github.com/Ramprasad-Group/canonicalize_psmiles
```

The traditional-ML benchmark (`ml-bench`) additionally needs `scikit-learn scipy rdkit torch transformers`, plus optionally `xgboost lightgbm catboost` (those three are skipped automatically at runtime if not installed). The polyBERT featurizer downloads the public checkpoint (`kuelumbus/polyBERT`) on first use.

## OpenAI API setup

Modes that submit fine-tune jobs or run inference need an OpenAI key, resolved in this order:

1. `openai_api_key.txt` in the working directory (one line; do not commit this file), or
2. the `OPENAI_API_KEY` environment variable.

No key is stored in the code. Paid modes (`train`, `retrieve`, `infer`, `autotune`, `all`) prompt for confirmation before spending; pass `--yes` to skip the prompt for non-interactive / tmux runs.

## Paths

Input/output paths default to the repo-relative locations above, so a fresh clone runs without editing the script. To point at another filesystem, set any of `LEO_DATASET_CSV`, `LEO_SPLITS_DIR`, `LEO_RESULTS_DIR`, `LEO_ML_OUT_DIR` (and `LEO_POLYBERT_PATH` for a local polyBERT cache).

## Pipeline modes

```
python scripts/leo_pipeline.py <mode> [--split-type {production,feature_ablation}] [--yes]
```

(`autotune` is its own mode, not a split type. The `autotune` split type applies only to `figures --split-type autotune`, which renders the sweep plots.)

| Mode        | Does                                                                        |
| ----------- | -------------------------------------------------------------------------- |
| `prep`      | Canonicalize pSMILES, build splits (CSV + JSONL). Idempotent.               |
| `train`     | Submit fine-tune jobs.                                                      |
| `retrieve`  | Poll jobs, save fine-tuned model ids and result files.                     |
| `infer`     | Run inference, aggregate combined CSVs.                                     |
| `figures`   | Generate parity plots.                                                      |
| `autotune`  | Resumable model / temperature / epoch sweep.                               |
| `all`       | `prep → train → retrieve → infer → figures`.                               |
| `ml-bench`  | Traditional-ML benchmark: feature ablation on the split, then 5-fold production; ranked CSV by chemistry-averaged OME. |

`--split-type` defaults to `production`.

### Quick start (traditional-ML benchmark, no API key)

```
python scripts/leo_pipeline.py ml-bench
```

Runs the ML benchmark end to end using only the local dataset and open featurizers (Morgan, RDKit descriptors, polyBERT) — no OpenAI calls.

### Full LLM run (production 5-fold)

```
python scripts/leo_pipeline.py all --split-type production --yes
```

Or step by step:

```
python scripts/leo_pipeline.py prep     --split-type production
python scripts/leo_pipeline.py train    --split-type production --yes
python scripts/leo_pipeline.py retrieve  --split-type production --yes
python scripts/leo_pipeline.py infer    --split-type production --yes
python scripts/leo_pipeline.py figures   --split-type production
```

### Feature ablation and autotune

```
python scripts/leo_pipeline.py all      --split-type feature_ablation --yes
python scripts/leo_pipeline.py autotune  --yes
```

## Outputs

Splits live under `splits/` (per-fold `train.csv` / `test.csv` / `.jsonl` plus manifests). LLM predictions, aggregated CSVs, and parity figures (SVG / EPS / PDF) land under `results/`; autotune writes its summary and best-config there. The ML benchmark writes to `results/ml_bench/`, including `ml_production_ranking.csv` (models ranked by chemistry-averaged OME with bootstrap CIs), per-model predictions, frozen hyperparameters, and run metadata.

The `splits/` and `results/` in this repo are the exact ones used in the paper — committed so the published results reproduce directly. Re-running any mode regenerates them in place (deterministically, from the same dataset CSV).

## Citation

If you use this dataset or pipeline, please cite the published work.
