# MODEL_COMPARISON — Ames Housing Regression (Task 3)

## 1. Objective

The objective was to predict `SalePrice` using a leakage-safe regression workflow and compare:

- A mean-prediction baseline (`DummyRegressor`)
- Plain Linear Regression
- Ridge Regression with four alpha values
- Raw-target models against models trained on `log1p(SalePrice)`

All metrics below are calculated on the same held-out test set and are reported in the original SalePrice scale.

## 2. Dataset and feature preparation

- Source: Task 1 cleaned Ames Housing dataset (`ames_housing_cleaned.csv`)
- Dataset size: **1,460 rows × 89 columns**
- Target: `SalePrice`
- Predictors retained: **84**
- Numeric predictors: **39**
- Categorical predictors: **45**
- `Id` was excluded because it is an identifier, not a meaningful property measurement.
- `SalePrice` was kept separate as the target.
- Three Task 1 IQR flags (`GrLivArea_outlier`, `LotArea_outlier`, `GarageArea_outlier`) were excluded. Their thresholds had been calculated using the full dataset, before this task's split, so they were not used as model features.
- `MSSubClass` and `MoSold` were treated as categorical variables, following the Task 1 data-type decisions.

### Important leakage correction from Task 1

The Task 1 cleaned CSV already contained some values imputed using full-dataset statistics. The saved row-level missing indicators made it possible to restore the original missing entries before this task's split:

- `LotFrontage`: 259 values
- `Electrical`: 1 value
- `MasVnrType`: 8 values
- `MasVnrArea`: 8 values
- `GarageYrBlt`: 81 values were still missing

This restored **357 missing cells**. Meaningful structural categories such as `"None"` were retained where they represented a feature that did not exist. No imputation statistics were calculated during this restoration step.

## 3. Train/test split

The split was performed before fitting any imputer, scaler, or encoder:

- Training set: **1,168 rows (80%)**
- Test set: **292 rows (20%)**
- `random_state=42`

A single split was used, as required by the task. Cross-validation was not used.

## 4. Preprocessing pipeline

The preprocessing steps were placed inside a scikit-learn `Pipeline` and `ColumnTransformer`:

| Feature type | Steps |
|---|---|
| Numeric | Median imputation → StandardScaler |
| Categorical | Most-frequent imputation → OneHotEncoder |

The encoder used `handle_unknown="ignore"` so a category appearing only in the test set would not crash prediction. `drop="first"` keeps one category as the reference level for each categorical feature.

The pipeline is fitted using `X_train` only. When `predict(X_test)` is called, the already-fitted training imputers, scaler, and encoder transform the test data. They are not refitted on the test set.

## 5. Test-set comparison

| Model | Target used for training | Alpha | MAE | RMSE | R² |
|---|---|---:|---:|---:|---:|
| Mean baseline | SalePrice | — | 62,575.93 | 87,619.03 | -0.0009 |
| Linear Regression | SalePrice | — | 19,786.88 | 40,657.67 | 0.7845 |
| Ridge Regression | SalePrice | 0.1 | 19,686.25 | 32,512.93 | 0.8622 |
| Ridge Regression | SalePrice | 1.0 | 20,056.82 | 30,849.07 | 0.8759 |
| Ridge Regression | SalePrice | 10.0 | 19,002.93 | 30,745.47 | 0.8768 |
| Ridge Regression | SalePrice | 100.0 | 19,067.99 | 32,896.91 | 0.8589 |
| Linear Regression | log1p(SalePrice) | — | 15,669.78 | 25,397.18 | 0.9159 |
| Ridge Regression | log1p(SalePrice) | 0.1 | 16,030.55 | 24,920.50 | 0.9190 |
| Ridge Regression | log1p(SalePrice) | 1.0 | 16,481.10 | 25,200.14 | 0.9172 |
| Ridge Regression | log1p(SalePrice) | 10.0 | 16,520.90 | 25,664.54 | 0.9141 |
| Ridge Regression | log1p(SalePrice) | 100.0 | 17,345.85 | 28,009.20 | 0.8977 |

For log-target models, predictions were converted back to the original SalePrice scale using `expm1` before calculating metrics. This makes the table directly comparable.

## 6. Interpretation

### Mean baseline

The mean baseline achieved an MAE of **62,575.93**, RMSE of **87,619.03**, and R² of **-0.0009**. This is a weak reference point: predicting one constant average price does not capture the differences between properties. The slightly negative R² means it performed marginally worse than predicting the test-set mean.

### Linear Regression vs Ridge on the original target

Plain Linear Regression achieved an RMSE of **40,657.67** and R² of **0.7845**.

The best raw-target Ridge result in this alpha comparison was **alpha = 10**, with RMSE **30,745.47** and R² **0.8768**. This is a meaningful improvement in RMSE and R², although the MAE improvement is more modest. Ridge regularization helps control large or unstable coefficients when predictors are correlated or when one-hot encoding creates many features.

### Effect of the log target

The training target had a right-skewness of **1.743**. After applying `log1p`, its skewness fell to **0.125**, making the training target much closer to symmetric.

The log-target Linear Regression achieved MAE **15,669.78**, RMSE **25,397.18**, and R² **0.9159**. The best log-target Ridge result by RMSE/R² was **alpha = 0.1**, with MAE **16,030.55**, RMSE **24,920.50**, and R² **0.9190**.

The log transformation improved performance substantially compared with the raw-target Linear Regression. On the log target, Ridge only slightly improved RMSE and R² over plain Linear Regression, while its MAE was slightly higher. Therefore, Ridge is not an unambiguous winner over plain Linear Regression after the log transformation: the preferred model depends on which metric matters most.

### Overall result

- Best MAE: **Log-target Linear Regression** — 15,669.78
- Best RMSE and R²: **Log-target Ridge, alpha = 0.1** — RMSE 24,920.50, R² 0.9190
- Both clearly outperform the mean baseline.

This is a single 80/20 split, so these results are a baseline estimate rather than a final production-quality performance claim.

## 7. What the metrics communicate

- **MAE (Mean Absolute Error):** Average absolute prediction error in SalePrice units. It is easy to explain as an average price error.
- **RMSE (Root Mean Squared Error):** Also in SalePrice units, but it penalizes large errors more strongly. It can be higher when a model makes a few very large mistakes.
- **R²:** Measures how much variation in SalePrice is explained by the model relative to a mean-prediction reference. It can be negative when a model is worse than the reference.

The metrics can disagree. In this experiment, log-target Linear Regression had the lowest MAE, while log-target Ridge had the lowest RMSE and highest R².

## 8. Conclusion

The mean baseline is substantially weaker than the trained models. Ridge provides a clear improvement over plain Linear Regression when both use the original SalePrice target. However, applying a log transformation to the skewed target gives a larger overall improvement. With the log target, Linear Regression and Ridge are close: Linear Regression has the better MAE, while Ridge has slightly better RMSE and R².

The most important achievement of this task is the leakage-safe workflow: split first, fit preprocessing only on training data, and apply the same fitted transformations to the test data.
