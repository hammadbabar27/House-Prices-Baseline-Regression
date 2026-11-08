# Task 3 — House Prices Regression Baseline

## Files

- `regression_baseline.py` — complete leakage-safe training and evaluation script.
- `MODEL_COMPARISON.md` — results table and interpretation.
- `model_comparison_results.csv` — machine-readable metrics table.
- `requirements.txt` — required Python packages.

## Dataset

Use the `ames_housing_cleaned.csv` file from Task 1.

## Run locally

```bash
pip install -r requirements.txt
python regression_baseline.py
```

The script prints the test metrics and saves `model_comparison_results.csv`.

## Run in Google Colab

1. Upload `regression_baseline.py` and `ames_housing_cleaned.csv` to the Colab session.
2. Run:

```python
!pip install pandas numpy scikit-learn
!python regression_baseline.py
```

3. Download `model_comparison_results.csv` from the Colab Files panel if you need it separately.

## Important

The script restores missing values that Task 1 had imputed using full-dataset statistics by using Task 1's saved missingness indicators. The train/test split then happens before the new imputers, scaler, and encoder are fitted. This is done to reduce leakage from the earlier cleaning stage.

My Task 1 Link:
https://github.com/hammadbabar27/Data-Cleaning-Quality-Audit

