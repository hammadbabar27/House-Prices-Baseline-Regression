"""Task 3 - Leakage-safe Ames Housing regression baseline."""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. Load the Task 1 cleaned dataset.
# Keep the real text category "None". Only Kaggle's "NA" and blank cells
# should be interpreted as missing values.
df = pd.read_csv(
    "ames_housing_cleaned.csv",
    keep_default_na=False,
    na_values=["NA", ""],
)

# SalePrice is the target and cannot be imputed.
df = df.dropna(subset=["SalePrice"]).copy()
y = df["SalePrice"].astype(float)


# 2. Restore values that Task 1 had imputed using full-dataset statistics.
# Task 1 saved row-level missing indicators, so we can restore the original
# missing entries without calculating any statistics from the full dataset.
for value_column, flag_column in [
    ("LotFrontage", "LotFrontage_missing"),
    ("Electrical", "Electrical_missing"),
]:
    if flag_column in df.columns:
        df.loc[df[flag_column].eq(1), value_column] = np.nan

# Preserve the meaningful "None" category when zero area confirms no veneer.
# Otherwise, restore the uncertain values so the pipeline imputes them.
if "MasVnrType_missing" in df.columns:
    no_veneer = df["MasVnrArea"].eq(0)
    restore_type = df["MasVnrType_missing"].eq(1) & ~no_veneer
    df.loc[restore_type, "MasVnrType"] = np.nan

if "MasVnrArea_missing" in df.columns:
    no_veneer_type = df["MasVnrType"].eq("None")
    restore_area = df["MasVnrArea_missing"].eq(1) & ~no_veneer_type
    df.loc[restore_area, "MasVnrArea"] = np.nan


# 3. Prepare features.
# Exclude SalePrice (target), Id (identifier), and IQR flags. The IQR flags
# were created from full-dataset thresholds in Task 1, so they are not used
# as predictors in this leakage-safe evaluation.
columns_to_drop = [
    "SalePrice",
    "Id",
    "GrLivArea_outlier",
    "LotArea_outlier",
    "GarageArea_outlier",
]
X = df.drop(columns=[c for c in columns_to_drop if c in df.columns]).copy()

# Reapply Task 1 data-type decisions that are not preserved by CSV files.
# These are numeric codes/labels, not continuous measurements.
for column in ["MSSubClass", "MoSold"]:
    if column in X.columns:
        X[column] = X[column].astype("category")


# 4. Split BEFORE fitting any imputer, scaler, or encoder.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

# Identify columns using the training data's data types.
numeric_columns = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_columns = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()


# 5. Build a fresh preprocessing + model pipeline for each experiment.
def make_preprocessor():
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def make_model_pipeline(model):
    return Pipeline(
        steps=[
            ("preprocessing", make_preprocessor()),
            ("model", model),
        ]
    )


def calculate_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }


# 6. Train the mean-prediction baseline.
results = []
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)

results.append(
    {
        "Model": "Mean baseline",
        "Target": "SalePrice",
        "Alpha": "—",
        **calculate_metrics(y_test, baseline_predictions),
    }
)


# 7. Train plain Linear Regression on the original SalePrice target.
linear_model = make_model_pipeline(LinearRegression())
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)

results.append(
    {
        "Model": "Linear Regression",
        "Target": "SalePrice",
        "Alpha": "—",
        **calculate_metrics(y_test, linear_predictions),
    }
)


# 8. Compare Ridge using four alpha values on the original target.
alpha_values = [0.1, 1.0, 10.0, 100.0]

for alpha in alpha_values:
    ridge_model = make_model_pipeline(Ridge(alpha=alpha))
    ridge_model.fit(X_train, y_train)
    ridge_predictions = ridge_model.predict(X_test)

    results.append(
        {
            "Model": "Ridge Regression",
            "Target": "SalePrice",
            "Alpha": alpha,
            **calculate_metrics(y_test, ridge_predictions),
        }
    )


# 9. Test a log-transformed target.
# SalePrice is positive, so log1p is safe. Predictions are converted back
# with expm1 BEFORE calculating MAE, RMSE, and R2. Therefore, all metrics
# are in the original SalePrice scale.
y_train_log = np.log1p(y_train)

log_linear_model = make_model_pipeline(LinearRegression())
log_linear_model.fit(X_train, y_train_log)
log_linear_predictions = np.expm1(log_linear_model.predict(X_test))

results.append(
    {
        "Model": "Linear Regression",
        "Target": "log1p(SalePrice)",
        "Alpha": "—",
        **calculate_metrics(y_test, log_linear_predictions),
    }
)

for alpha in alpha_values:
    log_ridge_model = make_model_pipeline(Ridge(alpha=alpha))
    log_ridge_model.fit(X_train, y_train_log)
    log_ridge_predictions = np.expm1(log_ridge_model.predict(X_test))

    results.append(
        {
            "Model": "Ridge Regression",
            "Target": "log1p(SalePrice)",
            "Alpha": alpha,
            **calculate_metrics(y_test, log_ridge_predictions),
        }
    )


# 10. Display and save the comparison table.
comparison = pd.DataFrame(results)
comparison.to_csv("model_comparison_results.csv", index=False)

print("\nDATA AND SPLIT")
print("Original cleaned dataset shape:", df.shape)
print("Feature count:", X.shape[1])
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Numeric features:", len(numeric_columns))
print("Categorical features:", len(categorical_columns))
print("Missing values restored before split:", int(X.isna().sum().sum()))
print("Training SalePrice skewness:", round(y_train.skew(), 3))
print("Training log1p(SalePrice) skewness:", round(y_train_log.skew(), 3))

print("\nTEST-SET MODEL COMPARISON")
print(
    comparison.to_string(
        index=False,
        formatters={
            "MAE": lambda value: f"{value:,.2f}",
            "RMSE": lambda value: f"{value:,.2f}",
            "R2": lambda value: f"{value:.4f}",
        },
    )
)

print("\nSaved: model_comparison_results.csv")
