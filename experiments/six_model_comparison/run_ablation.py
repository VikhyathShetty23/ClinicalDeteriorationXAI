
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
)

from xgboost import XGBClassifier


# ==================================================
# 1. PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = Path(
    r"C:\Users\Vikhyath Shetty\Downloads\Sepsis_Dataset\prepared_features.csv"
)

ORIGINAL_MODEL_PATH = BASE_DIR / "saved_models" / "original_xgboost.joblib"

COMMON_TEST_PATH = BASE_DIR / "results" / "common_test_predictions.csv"

OUTPUT_DIR = BASE_DIR / "ablation"

MODEL_DIR = OUTPUT_DIR / "saved_models"
RESULT_DIR = OUTPUT_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET = "SepsisLabel"


# ==================================================
# 2. LOAD DATA AND ORIGINAL FEATURE CONFIGURATION
# ==================================================

print("Loading prepared dataset...")

df = pd.read_csv(DATA_PATH)

original_artifact = joblib.load(ORIGINAL_MODEL_PATH)

original_features = original_artifact["feature_columns"]

print("Original feature count:", len(original_features))

required_columns = {
    "PatientID", "ICULOS", "Split", TARGET
}

if not required_columns.issubset(df.columns):
    raise ValueError("Prepared dataset is missing required columns.")

if df["PatientID"].isna().any():
    raise ValueError("PatientID contains missing values.")

# Confirm patient-level split integrity
patient_splits = df.groupby("PatientID")["Split"].nunique()

if (patient_splits > 1).any():
    raise ValueError("Patient leakage detected across splits.")

train_df = df[df["Split"] == "train"].copy()
val_df = df[df["Split"] == "validation"].copy()
test_df = df[df["Split"] == "test"].copy()

print("Training rows:", len(train_df))
print("Validation rows:", len(val_df))
print("Full test rows:", len(test_df))


# ==================================================
# 3. SELECT COMMON TEST POPULATION
# ==================================================

print("\nLoading common test endpoints...")

common_df = pd.read_csv(COMMON_TEST_PATH)

required_common = {"PatientID", "ICULOS", TARGET}

if not required_common.issubset(common_df.columns):
    raise ValueError(
        "Common test predictions file must contain "
        "PatientID, ICULOS and SepsisLabel."
    )

# Normalize join keys
for frame in (test_df, common_df):
    frame["PatientID"] = frame["PatientID"].astype(str)
    frame["ICULOS"] = pd.to_numeric(
        frame["ICULOS"], errors="raise"
    )

# Keep only rows represented in the existing common test set
common_keys = common_df[
    ["PatientID", "ICULOS", TARGET]
].drop_duplicates()

if common_keys.duplicated(["PatientID", "ICULOS"]).any():
    raise ValueError("Duplicate patient-hour keys in common test data.")

common_test = test_df.merge(
    common_keys,
    on=["PatientID", "ICULOS"],
    how="inner",
    suffixes=("", "_common"),
    validate="one_to_one"
)

if len(common_test) != len(common_keys):
    raise ValueError(
        "Could not match every common test endpoint. "
        f"Matched {len(common_test)} of {len(common_keys)}."
    )

if not (
    common_test[TARGET].to_numpy()
    == common_test[f"{TARGET}_common"].to_numpy()
).all():
    raise ValueError("Sepsis labels do not match between datasets.")

common_test = common_test.drop(columns=[f"{TARGET}_common"])

print("Common test endpoints:", len(common_test))
print("Common test positive labels:", int(common_test[TARGET].sum()))
print(
    "Common test unique patients:",
    common_test["PatientID"].nunique()
)


# ==================================================
# 4. THRESHOLD SELECTION
# ==================================================

def select_threshold(y_true, scores):

    precision, recall, thresholds = precision_recall_curve(
        y_true, scores
    )

    if len(thresholds) == 0:
        return 0.5

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    return float(thresholds[int(np.argmax(f1_values))])


# ==================================================
# 5. EVALUATION
# ==================================================

def evaluate_model(name, y_true, scores, threshold):

    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "Model": name,
        "Threshold": float(threshold),
        "AUROC": float(roc_auc_score(y_true, scores)),
        "AUPRC": float(average_precision_score(y_true, scores)),
        "Precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "Recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "F1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


# ==================================================
# 6. DEFINE ABLATION CONFIGURATIONS
# ==================================================

ablation_configs = {
    "XGBoost Without ICULOS": [
        col for col in original_features
        if col != "ICULOS"
    ],

    "XGBoost Without HospAdmTime": [
        col for col in original_features
        if col != "HospAdmTime"
    ],

    "XGBoost Without Both": [
        col for col in original_features
        if col not in ["ICULOS", "HospAdmTime"]
    ],
}


# ==================================================
# 7. TRAIN AND EVALUATE EACH MODEL
# ==================================================

positive = int(train_df[TARGET].sum())
negative = int(len(train_df) - positive)

scale_pos_weight = negative / positive

print("\nTraining positive labels:", positive)
print("Training negative labels:", negative)
print("Scale positive weight:", round(scale_pos_weight, 4))


def train_ablation_model(name, columns):

    print("\n" + "=" * 65)
    print("Training:", name)
    print("Number of features:", len(columns))
    print("=" * 65)

    X_train = train_df[columns].copy()
    X_val = val_df[columns].copy()
    X_test = common_test[columns].copy()

    y_train = train_df[TARGET].astype(int).to_numpy()
    y_val = val_df[TARGET].astype(int).to_numpy()
    y_test = common_test[TARGET].astype(int).to_numpy()

    # Fit imputer only on training data
    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True
    )

    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    # Same XGBoost configuration as original training
    model = XGBClassifier(
        n_estimators=700,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=2,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    print("Fitting model...")
    model.fit(X_train, y_train)

    # Validation scores determine threshold
    val_scores = model.predict_proba(X_val)[:, 1]

    threshold = select_threshold(y_val, val_scores)

    # Evaluate on common test endpoints
    test_scores = model.predict_proba(X_test)[:, 1]

    result = evaluate_model(
        name,
        y_test,
        test_scores,
        threshold
    )

    # Save model and its fitted imputer separately
    artifact = {
        "model": model,
        "imputer": imputer,
        "scaler": None,
        "feature_columns": columns,
        "threshold": threshold,
        "model_name": name,
    }

    filename = (
        name.lower()
        .replace(" ", "_")
    )

    joblib.dump(
        artifact,
        MODEL_DIR / f"{filename}.joblib"
    )

    print("Validation threshold:", round(threshold, 5))
    print("Test AUROC:", round(result["AUROC"], 4))
    print("Test AUPRC:", round(result["AUPRC"], 4))
    print("Test Precision:", round(result["Precision"], 4))
    print("Test Recall:", round(result["Recall"], 4))
    print("Test F1:", round(result["F1"], 4))

    return result


# ==================================================
# 8. RUN ALL THREE ABLATION EXPERIMENTS
# ==================================================

results = []

for name, columns in ablation_configs.items():
    result = train_ablation_model(name, columns)
    results.append(result)


# ==================================================
# 9. SAVE RESULTS
# ==================================================

results_df = pd.DataFrame(results)

results_path = RESULT_DIR / "ablation_comparison.csv"

results_df.to_csv(results_path, index=False)

print("\n" + "=" * 65)
print("ABLATION EXPERIMENT COMPLETED")
print("=" * 65)

print(
    results_df[
        [
            "Model",
            "AUROC",
            "AUPRC",
            "Precision",
            "Recall",
            "F1"
        ]
    ].to_string(index=False)
)

print("\nModels saved in:", MODEL_DIR)
print("Results saved in:", results_path)
