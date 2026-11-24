
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ==================================================
# PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = Path(
    r"C:\Users\Vikhyath Shetty\Downloads\Sepsis_Dataset\prepared_features.csv"
)

OUTPUT_DIR = Path(__file__).resolve().parent
MODEL_DIR = OUTPUT_DIR / "saved_models"
RESULT_DIR = OUTPUT_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET = "SepsisLabel"


# ==================================================
# LOAD DATA
# ==================================================

print("Loading prepared dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

required = {"PatientID", "Split", TARGET}

if not required.issubset(df.columns):
    raise ValueError("Dataset is missing required columns.")

if df["PatientID"].isna().any():
    raise ValueError("PatientID contains missing values.")

if df["Split"].isna().any():
    raise ValueError("Split contains missing values.")

# Feature columns exclude identifiers, split and target
feature_columns = [
    col for col in df.columns
    if col not in ["PatientID", "Split", TARGET]
]

clinical_columns = [
    col for col in feature_columns
    if col not in ["ICULOS", "HospAdmTime"]
]

if len(feature_columns) != 48:
    raise ValueError(
        f"Expected 48 features, found {len(feature_columns)}"
    )

if len(clinical_columns) != 46:
    raise ValueError(
        f"Expected 46 clinical features, found {len(clinical_columns)}"
    )

# Check patient separation
patient_splits = df.groupby("PatientID")["Split"].nunique()

if (patient_splits > 1).any():
    raise ValueError("Patient leakage detected across splits.")

train_df = df[df["Split"] == "train"].copy()
val_df = df[df["Split"] == "validation"].copy()
test_df = df[df["Split"] == "test"].copy()

print("\nSplit sizes:")
print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)

y_train = train_df[TARGET].astype(int).to_numpy()
y_val = val_df[TARGET].astype(int).to_numpy()
y_test = test_df[TARGET].astype(int).to_numpy()

positive = int(y_train.sum())
negative = int(len(y_train) - positive)

scale_pos_weight = negative / positive

print("\nTraining positive labels:", positive)
print("Training negative labels:", negative)
print("Scale positive weight:", round(scale_pos_weight, 2))


# ==================================================
# THRESHOLD SELECTION
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

    best_index = int(np.argmax(f1_values))

    return float(thresholds[best_index])


# ==================================================
# EVALUATION
# ==================================================

def evaluate_model(name, y_true, scores, threshold):

    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
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
        "TP": int(tp)
    }


# ==================================================
# MODEL TRAINING FUNCTION
# ==================================================

results = []


def train_and_evaluate(
    name,
    estimator,
    columns,
    scale_data=False
):

    print("\n" + "=" * 60)
    print("Training:", name)
    print("=" * 60)

    X_train = train_df[columns].copy()
    X_val = val_df[columns].copy()
    X_test = test_df[columns].copy()

    # Fit imputer on training data only
    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True
    )

    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    scaler = None

    if scale_data:

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    print("Fitting model...")

    estimator.fit(X_train, y_train)

    # Use continuous scores for threshold selection and metrics
    if hasattr(estimator, "predict_proba"):
        val_scores = estimator.predict_proba(X_val)[:, 1]
        test_scores = estimator.predict_proba(X_test)[:, 1]
    else:
        val_scores = estimator.decision_function(X_val)
        test_scores = estimator.decision_function(X_test)

    # Threshold selected only on validation data
    threshold = select_threshold(y_val, val_scores)

    # Final metrics on untouched test data
    result = evaluate_model(
        name,
        y_test,
        test_scores,
        threshold
    )

    results.append(result)

    # Save model and preprocessing objects
    artifact = {
        "model": estimator,
        "imputer": imputer,
        "scaler": scaler,
        "feature_columns": columns,
        "threshold": threshold,
        "model_name": name
    }

    filename = name.lower().replace(" ", "_")

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
# 1. ORIGINAL XGBOOST - 48 FEATURES
# ==================================================

original_xgb = XGBClassifier(
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

train_and_evaluate(
    "Original XGBoost",
    original_xgb,
    feature_columns
)


# ==================================================
# 2. CLINICAL-ONLY XGBOOST - 46 FEATURES
# ==================================================

clinical_xgb = XGBClassifier(
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

train_and_evaluate(
    "Clinical Only XGBoost",
    clinical_xgb,
    clinical_columns
)


# ==================================================
# 3. RANDOM FOREST
# ==================================================

random_forest = RandomForestClassifier(
    n_estimators=300,
    max_depth=18,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=RANDOM_STATE
)

train_and_evaluate(
    "Random Forest",
    random_forest,
    feature_columns
)


# ==================================================
# 4. LOGISTIC REGRESSION
# ==================================================

logistic = LogisticRegression(
    max_iter=500,
    solver="lbfgs",
    class_weight="balanced",
    random_state=RANDOM_STATE
)

train_and_evaluate(
    "Logistic Regression",
    logistic,
    feature_columns,
    scale_data=True
)


# ==================================================
# 5. LINEAR SVM
# ==================================================

svm = LinearSVC(
    C=0.1,
    class_weight="balanced",
    max_iter=3000,
    random_state=RANDOM_STATE
)

train_and_evaluate(
    "Linear SVM",
    svm,
    feature_columns,
    scale_data=True
)


# ==================================================
# SAVE RESULTS
# ==================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULT_DIR / "tabular_model_comparison.csv",
    index=False
)

with open(
    RESULT_DIR / "tabular_model_comparison.json",
    "w"
) as f:
    json.dump(results, f, indent=4)

print("\n" + "=" * 60)
print("TABULAR MODEL TRAINING COMPLETED")
print("=" * 60)

print(
    results_df[
        ["Model", "AUROC", "AUPRC", "Precision", "Recall", "F1"]
    ].to_string(index=False)
)

print("\nSaved models:", MODEL_DIR)
print("Saved results:", RESULT_DIR)
