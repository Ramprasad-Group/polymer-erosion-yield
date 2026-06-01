# polymer-erosion-yield
Curated polymer atomic oxygen erosion yield dataset and the scripts used to fine-tune and evaluate the LLM-based erosion yield predictor.

## Repository layout

```
polymer-erosion-yield/
├── data/
│   ├── polymer_Ey_dataset_current.csv        # actively maintained current dataset
│   └── polymer_ey_dataset_paper_v1_0_0.csv   # static paper-specific release
└── scripts/
    ├── data_processing.py                    # CSV → JSONL conversion for fine-tuning
    ├── finetune_and_test.py                  # fetch fine-tuned model + run inference
    └── combined_orient_plot.py               # parity plots
```
## Workflow summary

The pipeline runs in three stages, from raw dataset to evaluated predictions:

1. **Data Processing**
   Convert each split's `train {N}.csv` / `test {N}.csv` into the OpenAI chat-completion JSONL format used for fine-tuning (`data_processing.py`).

2. **Fine-tuning & inference**
   Fetch the completed fine-tuned model, optionally diagnose the job, and run inference over the train/test splits, writing predictions to CSV (`finetune_and_test.py`).

3. **Evaluation & visualization**
   Aggregate the per-split result CSVs and render log–log parity plots with OME and log R² metrics (`combined_orient_plot.py`).

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
python scripts/combined_orient_plot.py
```
