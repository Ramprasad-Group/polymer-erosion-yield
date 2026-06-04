# polymer-erosion-yield
Curated polymer atomic oxygen erosion yield dataset and the scripts used to fine-tune and evaluate the LLM-based erosion yield predictor.

## Overview

This repository contains a curated dataset of polymer atomic oxygen (AO) erosion yields collected from multiple low Earth orbit (LEO) spaceflight experiments. The dataset combines polymer chemical structures, environmental exposure conditions, and experimentally measured erosion yields to support research in space materials, polymer degradation, and materials informatics.

## Repository layout

```
polymer-erosion-yield/
├── data/
│   ├── polymer_Ey_dataset_current.csv        # actively maintained current dataset
│   ├── polymer_ey_dataset_paper_v1_0_0.csv   # static paper-specific release
│   ├── GPR random split Ey.csv               # GPR Ey predictions using the random split
│   └── GPR RG split Ey.csv                   # GPR Ey predictions using the restricted group split
└── scripts/
    ├── data_processing.py                    # CSV → JSONL conversion for fine-tuning
    ├── finetune_and_test.py                  # fetch fine-tuned model + run inference
    └── production_figures.py                 # parity plots and comparison with GPR
```

## Dataset

| File                             | Description                            |
| -------------------------------- | -------------------------------------- |
| `polymer_ey_dataset_paper_v1_0_0.csv` | Complete dataset used in the paper |
| `polymer_Ey_dataset_current.csv` | Complete extracted polymer erosion yield dataset |

### Dataset Statistics

* **Total samples:** 195
* **Number of features:** 11
* **Target property:** Atomic oxygen erosion yield

### Column Description

| Column                       | Description                               |
| ---------------------------- | ----------------------------------------- |
| `polymer name`               | Polymer name                              |
| `coating name`               | Coating applied to the material (if any)  |
| `smiles`                     | Polymer repeat-unit SMILES representation |
| `layers`                     | Number of layers                          |
| `thickness (mils)`           | Thickness in mils                         |
| `thickness (mm)`             | Thickness in millimeters                  |
| `e_y (cm3/atom)`             | Erosion yield in cm³/atom                 |
| `e_y (A3/atom)`              | Erosion yield in Å³/atom                  |
| `log(e_y)`                   | Log-transformed erosion yield             |
| `mission_time (yr)`          | Exposure duration in years                |
| `ram_solar_exposure (esh)`   | Equivalent Sun Hours (ESH)                |
| `ram_ao_fluence (atoms/cm2)` | Atomic oxygen fluence                     |
| `mission`                    | Space mission source                      |
| `orientation`                | Exposure orientation                      |

### Target Property

The primary target variable is the log atomic oxygen erosion yield (`log(e_y)`), which quantifies the volume of material removed per incident oxygen atom. Both raw and log-transformed values are provided in the dataset.

### Data Source

The dataset was compiled from multiple LEO spaceflight experiments and missions, including MISSE, EOIM, LDEF, STS, and related exposure studies.

## Workflow summary

The pipeline runs in three stages, from raw dataset to evaluated predictions:

1. **Data Processing**
   Convert each split's `train {N}.csv` / `test {N}.csv` into the OpenAI chat-completion JSONL format used for fine-tuning (`data_processing.py`).

2. **Fine-tuning & inference**
   Fetch the completed fine-tuned model, optionally diagnose the job, and run inference over the train/test splits, writing predictions to CSV (`finetune_and_test.py`).

3. **Evaluation & visualization**
   Aggregate the per-split result CSVs and render log–log parity plots with OME and log R² metrics (`production_figures.py`).

## Installation

```
pip install openai pandas numpy matplotlib
```

## OpenAI API setup

`scripts/finetune_and_test.py` calls fine-tuned OpenAI models. Set your API key by editing the `api_key` placeholder at the top of the script.

Do **not** commit API keys or generated results to GitHub.

## Usage

Set the path/index variables at the top of each script (`INPUT_DIR`, `OUTPUT_DIR`, `N`, and `BASE_DIR` / `SPLIT_COUNT` for plotting), then run:

```
# 1. Format each split's CSVs into JSONL
python scripts/data_processing.py

# 2. Fetch the fine-tuned model and run inference
python scripts/finetune_and_test.py

# 3. Aggregate results and generate parity plots
python scripts/production_figures.py
```

## Citation

If you use this dataset in your research, please cite the published work.
