
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "six_model_comparison"
    / "saved_models"
    / "original_xgboost.joblib"
)

DATA_PATH = Path(
    r"C:\Users\Vikhyath Shetty\Downloads\Sepsis_Dataset\prepared_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "six_model_comparison"
    / "results"
    / "shap"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 1000
RANDOM_STATE = 42


# --------------------------------------------------
# 2. Load saved model and preprocessing objects
# --------------------------------------------------

print("Loading saved Original XGBoost model...")

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
imputer = artifact["imputer"]
scaler = artifact["scaler"]
feature_columns = artifact["feature_columns"]

print("Model loaded:", artifact["model_name"])
print("Number of features:", len(feature_columns))
print("Classification threshold:", artifact["threshold"])


# --------------------------------------------------
# 3. Load prepared dataset and select test patients
# --------------------------------------------------

print("\nLoading prepared dataset...")

df = pd.read_csv(DATA_PATH)

test_df = df[df["Split"].str.lower() == "test"].copy()

if test_df.empty:
    raise ValueError("No test rows found. Check the Split column.")

X_test = test_df[feature_columns]
y_test = test_df["SepsisLabel"]

print("Test rows:", len(X_test))
print("Test positive labels:", int(y_test.sum()))


# --------------------------------------------------
# 4. Apply saved preprocessing
# --------------------------------------------------

print("\nApplying saved preprocessing...")

X_imputed = imputer.transform(X_test)

if scaler is not None:
    X_processed = scaler.transform(X_imputed)
else:
    X_processed = X_imputed

X_processed = pd.DataFrame(
    X_processed,
    columns=feature_columns,
    index=X_test.index
)

print("Preprocessed test shape:", X_processed.shape)


# --------------------------------------------------
# 5. Select a reproducible sample
# --------------------------------------------------

sample_size = min(SAMPLE_SIZE, len(X_processed))

X_sample = X_processed.sample(
    n=sample_size,
    random_state=RANDOM_STATE
)

print("\nSHAP sample size:", len(X_sample))


# --------------------------------------------------
# 6. Calculate SHAP values
# --------------------------------------------------

print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(model)

shap_explanation = explainer(
    X_sample,
    check_additivity=False
)

# Handle possible multi-output SHAP values
if shap_explanation.values.ndim == 3:
    values = shap_explanation.values[:, :, 1]

    base_values = shap_explanation.base_values

    if np.ndim(base_values) == 2:
        base_values = base_values[:, 1]

    shap_explanation = shap.Explanation(
        values=values,
        base_values=base_values,
        data=X_sample.to_numpy(),
        feature_names=feature_columns
    )

print("SHAP calculation completed.")
print("SHAP shape:", shap_explanation.values.shape)


# --------------------------------------------------
# 7. Global feature importance
# --------------------------------------------------

importance = pd.DataFrame({
    "Feature": feature_columns,
    "Mean_Absolute_SHAP": np.abs(
        shap_explanation.values
    ).mean(axis=0)
})

importance = importance.sort_values(
    by="Mean_Absolute_SHAP",
    ascending=False
)

importance.to_csv(
    OUTPUT_DIR / "shap_feature_importance.csv",
    index=False
)

print("\nTop 15 important features:")
print(importance.head(15).to_string(index=False))


# --------------------------------------------------
# 8. SHAP summary / beeswarm plot
# --------------------------------------------------

print("\nGenerating SHAP summary plot...")

shap.plots.beeswarm(
    shap_explanation,
    max_display=20,
    show=False
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "shap_summary_beeswarm.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 9. SHAP bar plot
# --------------------------------------------------

print("Generating SHAP bar plot...")

shap.plots.bar(
    shap_explanation,
    max_display=20,
    show=False
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 10. Individual prediction waterfall plot
# --------------------------------------------------

print("Generating SHAP waterfall plot...")

shap.plots.waterfall(
    shap_explanation[0],
    max_display=15,
    show=False
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "shap_waterfall_example.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 11. Completion
# --------------------------------------------------

print("\nSHAP analysis completed successfully!")

print("Results saved in:")
print(OUTPUT_DIR)

print("\nGenerated files:")
for file in OUTPUT_DIR.iterdir():
    print("-", file.name)
