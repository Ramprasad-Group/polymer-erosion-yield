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
    └── combined_orient_plot.py               # parity plots by orientation
```
