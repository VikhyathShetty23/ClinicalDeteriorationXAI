
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ==================================================
# PATHS AND CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = Path(
    r"C:\Users\Vikhyath Shetty\Downloads\Sepsis_Dataset\prepared_features.csv"
)

MODEL_DIR = BASE_DIR / "saved_models"
RESULT_DIR = BASE_DIR / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEQUENCE_LENGTH = 6

LSTM_FEATURES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "Resp",
    "HR_missing", "O2Sat_missing", "Temp_missing",
    "SBP_missing", "MAP_missing", "Resp_missing",
    "Age", "Gender"
]

TARGET = "SepsisLabel"


# ==================================================
# LSTM ARCHITECTURE
# ==================================================

class TemporalAttentionLSTM(nn.Module):

    def __init__(self, input_size=14, hidden_size=32):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.attention = nn.Linear(hidden_size, 1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1)
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        attention_scores = self.attention(output)
        attention_weights = torch.softmax(attention_scores, dim=1)

        context = torch.sum(
            attention_weights * output,
            dim=1
        )

        logits = self.classifier(context)

        return logits.squeeze(1), attention_weights.squeeze(-1)


# ==================================================
# BUILD COMMON ELIGIBLE HOURLY OBSERVATIONS
# ==================================================

def build_eligible_data(data):

    endpoint_rows = []
    sequences = []

    for patient_id, group in data.groupby("PatientID", sort=False):

        group = group.sort_values("ICULOS").reset_index(drop=True)

        hours = group["ICULOS"].to_numpy()
        values = group[LSTM_FEATURES].to_numpy(dtype=np.float32)

        for end in range(SEQUENCE_LENGTH - 1, len(group)):

            start = end - SEQUENCE_LENGTH + 1

            window_hours = hours[start:end + 1]

            if not np.all(np.diff(window_hours) == 1):
                continue

            endpoint_rows.append(group.iloc[end])
            sequences.append(values[start:end + 1])

    if not sequences:
        raise ValueError("No eligible six-hour sequences found.")

    eligible = pd.DataFrame(endpoint_rows).reset_index(drop=True)

    X_sequences = np.asarray(sequences, dtype=np.float32)

    return eligible, X_sequences


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

    return float(thresholds[np.argmax(f1_values)])


# ==================================================
# METRICS
# ==================================================

def calculate_metrics(name, y_true, scores, threshold):

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
        "TP": int(tp)
    }


# ==================================================
# LOAD DATA AND BUILD COMMON POPULATION
# ==================================================

print("Loading prepared dataset...")

df = pd.read_csv(DATA_PATH)

train_df = df[df["Split"] == "train"].copy()
val_df = df[df["Split"] == "validation"].copy()
test_df = df[df["Split"] == "test"].copy()

print("\nBuilding eligible validation observations...")
val_eligible, X_val_sequences = build_eligible_data(val_df)

print("\nBuilding eligible test observations...")
test_eligible, X_test_sequences = build_eligible_data(test_df)

y_val = val_eligible[TARGET].astype(int).to_numpy()
y_test = test_eligible[TARGET].astype(int).to_numpy()

print("\nCOMMON EVALUATION POPULATION")
print("--------------------------------")
print("Validation observations:", len(y_val))
print("Validation positive labels:", int(y_val.sum()))
print("Test observations:", len(y_test))
print("Test positive labels:", int(y_test.sum()))

# Verify every model will use the same patient-hour endpoints
endpoint_columns = ["PatientID", "ICULOS"]

print("\nChecking validation and test endpoint uniqueness...")

if val_eligible.duplicated(endpoint_columns).any():
    raise ValueError("Duplicate validation patient-hour endpoints found.")

if test_eligible.duplicated(endpoint_columns).any():
    raise ValueError("Duplicate test patient-hour endpoints found.")


# ==================================================
# LOAD AND EVALUATE FIVE TABULAR MODELS
# ==================================================

tabular_models = [
    "original_xgboost",
    "clinical_only_xgboost",
    "random_forest",
    "logistic_regression",
    "linear_svm"
]

all_results = []
all_predictions = test_eligible[endpoint_columns + [TARGET]].copy()


for model_key in tabular_models:

    artifact_path = MODEL_DIR / f"{model_key}.joblib"

    print("\nEvaluating:", model_key)

    artifact = joblib.load(artifact_path)

    model = artifact["model"]
    imputer = artifact["imputer"]
    scaler = artifact["scaler"]
    columns = artifact["feature_columns"]

    # Validation input
    X_val = val_eligible[columns].copy()
    X_val = imputer.transform(X_val)

    if scaler is not None:
        X_val = scaler.transform(X_val)

    # Test input
    X_test = test_eligible[columns].copy()
    X_test = imputer.transform(X_test)

    if scaler is not None:
        X_test = scaler.transform(X_test)

    # Continuous prediction scores
    if hasattr(model, "predict_proba"):
        val_scores = model.predict_proba(X_val)[:, 1]
        test_scores = model.predict_proba(X_test)[:, 1]
    else:
        val_scores = model.decision_function(X_val)
        test_scores = model.decision_function(X_test)

    # Select threshold using eligible validation endpoints
    threshold = select_threshold(y_val, val_scores)

    result = calculate_metrics(
        artifact["model_name"],
        y_test,
        test_scores,
        threshold
    )

    all_results.append(result)

    prediction_name = model_key + "_score"
    all_predictions[prediction_name] = test_scores

    print(
        f"AUROC: {result['AUROC']:.4f} | "
        f"AUPRC: {result['AUPRC']:.4f} | "
        f"F1: {result['F1']:.4f}"
    )


# ==================================================
# LOAD LSTM PREPROCESSING
# ==================================================

print("\nPreparing LSTM inputs...")

lstm_imputer = joblib.load(
    MODEL_DIR / "six_model_lstm_imputer.joblib"
)

lstm_scaler = joblib.load(
    MODEL_DIR / "six_model_lstm_scaler.joblib"
)

def preprocess_sequences(X):

    shape = X.shape

    X = lstm_imputer.transform(
        X.reshape(-1, len(LSTM_FEATURES))
    )

    X = lstm_scaler.transform(X)

    return X.reshape(shape).astype(np.float32)


X_val_lstm = preprocess_sequences(X_val_sequences)
X_test_lstm = preprocess_sequences(X_test_sequences)


# ==================================================
# LOAD LSTM CHECKPOINT
# ==================================================

checkpoint_path = (
    MODEL_DIR / "six_model_temporal_attention_lstm.pth"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=False
)

lstm_model = TemporalAttentionLSTM(
    input_size=checkpoint["input_size"],
    hidden_size=checkpoint["hidden_size"]
).to(DEVICE)

lstm_model.load_state_dict(checkpoint["state_dict"])
lstm_model.eval()


def lstm_predict(X, batch_size=512):

    scores = []

    with torch.no_grad():

        for start in range(0, len(X), batch_size):

            batch = torch.tensor(
                X[start:start + batch_size],
                dtype=torch.float32,
                device=DEVICE
            )

            logits, _ = lstm_model(batch)

            probabilities = torch.sigmoid(logits)

            scores.extend(probabilities.cpu().numpy())

    return np.asarray(scores)


# ==================================================
# LSTM VALIDATION AND TEST PREDICTIONS
# ==================================================

print("\nGenerating LSTM validation scores...")

val_lstm_scores = lstm_predict(X_val_lstm)

print("Generating LSTM test scores...")

test_lstm_scores = lstm_predict(X_test_lstm)

lstm_threshold = select_threshold(
    y_val,
    val_lstm_scores
)

lstm_result = calculate_metrics(
    "Temporal Attention LSTM",
    y_test,
    test_lstm_scores,
    lstm_threshold
)

all_results.append(lstm_result)

all_predictions["temporal_attention_lstm_score"] = test_lstm_scores


# ==================================================
# SAVE FINAL COMPARISON
# ==================================================

results_df = pd.DataFrame(all_results)

results_df = results_df.sort_values(
    "F1",
    ascending=False
).reset_index(drop=True)

results_path = RESULT_DIR / "common_test_model_comparison.csv"

results_df.to_csv(results_path, index=False)

all_predictions.to_csv(
    RESULT_DIR / "common_test_predictions.csv",
    index=False
)

print("\n" + "=" * 65)
print("COMMON TEST-SET MODEL COMPARISON")
print("=" * 65)

print(
    results_df[
        ["Model", "AUROC", "AUPRC", "Precision", "Recall", "F1"]
    ].to_string(index=False)
)

print("\nCommon test observations:", len(y_test))
print("Common test positive labels:", int(y_test.sum()))

print("\nSaved comparison:", results_path)
print("Saved predictions:", RESULT_DIR / "common_test_predictions.csv")

print("\nCOMMON EVALUATION COMPLETED SUCCESSFULLY.")
