
from pathlib import Path
import json
import random
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
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
# CONFIGURATION
# ==================================================

SEED = 42
SEQUENCE_LENGTH = 6
INPUT_SIZE = 14
HIDDEN_SIZE = 32
EPOCHS = 0
BATCH_SIZE = 512
LEARNING_RATE = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = Path(
    r"C:\Users\Vikhyath Shetty\Downloads\Sepsis_Dataset\prepared_features.csv"
)

MODEL_DIR = BASE_DIR / "saved_models"
RESULT_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# FEATURES
# ==================================================

FEATURES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "Resp",
    "HR_missing", "O2Sat_missing", "Temp_missing",
    "SBP_missing", "MAP_missing", "Resp_missing",
    "Age", "Gender"
]

TARGET = "SepsisLabel"


# ==================================================
# MODEL ARCHITECTURE
# Same architecture as existing application
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

        lstm_output, _ = self.lstm(x)

        attention_scores = self.attention(lstm_output)

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        context = torch.sum(
            attention_weights * lstm_output,
            dim=1
        )

        output = self.classifier(context)

        return output.squeeze(1), attention_weights.squeeze(-1)


# ==================================================
# BUILD PATIENT-SPECIFIC SEQUENCES
# ==================================================

def build_sequences(df):

    sequences = []
    labels = []

    patient_count = df["PatientID"].nunique()
    skipped_windows = 0

    for patient_id, group in df.groupby("PatientID", sort=False):

        group = group.sort_values("ICULOS")

        values = group[FEATURES].to_numpy(dtype=np.float32)
        hours = group["ICULOS"].to_numpy()
        targets = group[TARGET].to_numpy(dtype=np.int64)

        if len(group) < SEQUENCE_LENGTH:
            continue

        for end in range(SEQUENCE_LENGTH - 1, len(group)):

            start = end - SEQUENCE_LENGTH + 1

            # Ensure all six observations are consecutive
            window_hours = hours[start:end + 1]

            if not np.all(np.diff(window_hours) == 1):
                skipped_windows += 1
                continue

            sequences.append(values[start:end + 1])

            # Predict the label at the final hour
            labels.append(targets[end])

    X = np.asarray(sequences, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    print(
        f"Patients: {patient_count}, "
        f"Sequences: {len(X)}, "
        f"Positive sequences: {int(y.sum())}"
    )

    return X, y, skipped_windows


# ==================================================
# LOAD DATA
# ==================================================

print("\nLoading prepared features...")

df = pd.read_csv(DATA_PATH)

train_df = df[df["Split"] == "train"].copy()
val_df = df[df["Split"] == "validation"].copy()
test_df = df[df["Split"] == "test"].copy()

print("\nBuilding training sequences...")
X_train, y_train, skipped_train = build_sequences(train_df)

print("\nBuilding validation sequences...")
X_val, y_val, skipped_val = build_sequences(val_df)

print("\nBuilding test sequences...")
X_test, y_test, skipped_test = build_sequences(test_df)

print("\nSequence shapes:")
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

print("\nSkipped non-consecutive windows:")
print("Train:", skipped_train)
print("Validation:", skipped_val)
print("Test:", skipped_test)


# ==================================================
# PREPROCESSING
# Fit ONLY on training sequences
# ==================================================

print("\nFitting LSTM imputer...")

imputer = SimpleImputer(
    strategy="median",
    keep_empty_features=True
)

train_flat = X_train.reshape(-1, INPUT_SIZE)

imputer.fit(train_flat)

X_train = imputer.transform(
    train_flat
).reshape(-1, SEQUENCE_LENGTH, INPUT_SIZE)

X_val = imputer.transform(
    X_val.reshape(-1, INPUT_SIZE)
).reshape(-1, SEQUENCE_LENGTH, INPUT_SIZE)

X_test = imputer.transform(
    X_test.reshape(-1, INPUT_SIZE)
).reshape(-1, SEQUENCE_LENGTH, INPUT_SIZE)


print("Fitting LSTM scaler...")

scaler = StandardScaler()

scaler.fit(
    X_train.reshape(-1, INPUT_SIZE)
)

X_train = scaler.transform(
    X_train.reshape(-1, INPUT_SIZE)
).reshape(-1, SEQUENCE_LENGTH, INPUT_SIZE)

X_val = scaler.transform(
    X_val.reshape(-1, INPUT_SIZE)
).reshape(-1, SEQUENCE_LENGTH, INPUT_SIZE)

X_test = scaler.transform(
    X_test.reshape(-1, INPUT_SIZE)
).reshape(-1, SEQUENCE_LENGTH, INPUT_SIZE)

# Verify preprocessing
assert not np.isnan(X_train).any()
assert not np.isnan(X_val).any()
assert not np.isnan(X_test).any()

print("Preprocessing completed. No NaN values remain.")


# Save preprocessing objects separately
joblib.dump(
    imputer,
    MODEL_DIR / "six_model_lstm_imputer.joblib"
)

joblib.dump(
    scaler,
    MODEL_DIR / "six_model_lstm_scaler.joblib"
)


# ==================================================
# DATA LOADERS
# ==================================================

def make_loader(X, y, shuffle=False):

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )


train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader = make_loader(X_val, y_val)
test_loader = make_loader(X_test, y_test)


# ==================================================
# INITIALIZE MODEL
# ==================================================

model = TemporalAttentionLSTM(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE
).to(DEVICE)

positive = int(y_train.sum())
negative = len(y_train) - positive

pos_weight = torch.tensor(
    [negative / max(positive, 1)],
    dtype=torch.float32,
    device=DEVICE
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("\nTraining positive sequences:", positive)
print("Training negative sequences:", negative)
print("Positive class weight:", round(
    negative / max(positive, 1), 2
))


# ==================================================
# PREDICTION FUNCTION
# ==================================================

def get_scores(loader):

    model.eval()

    all_scores = []
    all_labels = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(DEVICE)

            logits, _ = model(X_batch)

            probabilities = torch.sigmoid(logits)

            all_scores.extend(
                probabilities.cpu().numpy()
            )

            all_labels.extend(
                y_batch.numpy()
            )

    return (
        np.asarray(all_labels, dtype=int),
        np.asarray(all_scores)
    )


# ==================================================
# TRAINING LOOP
# Select best model using validation AUPRC
# ==================================================

best_val_auprc = -1
best_epoch = 0

checkpoint_path = (
    MODEL_DIR / "six_model_temporal_attention_lstm.pth"
)

history = []

print("\nStarting LSTM training...")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        logits, _ = model(X_batch)

        loss = criterion(logits, y_batch)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * len(y_batch)

    train_loss = total_loss / len(y_train)

    val_labels, val_scores = get_scores(val_loader)

    val_auprc = average_precision_score(
        val_labels, val_scores
    )

    val_auroc = roc_auc_score(
        val_labels, val_scores
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "validation_auroc": val_auroc,
        "validation_auprc": val_auprc
    })

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {train_loss:.5f} | "
        f"Val AUROC: {val_auroc:.4f} | "
        f"Val AUPRC: {val_auprc:.4f}"
    )

    if val_auprc > best_val_auprc:

        best_val_auprc = val_auprc
        best_epoch = epoch + 1

        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_size": INPUT_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "sequence_length": SEQUENCE_LENGTH,
                "best_epoch": best_epoch,
                "validation_auprc": best_val_auprc
            },
            checkpoint_path
        )


# ==================================================
# LOAD BEST CHECKPOINT
# ==================================================

print("\nLoading best LSTM checkpoint...")

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=False
)

best_epoch = checkpoint["best_epoch"]

model.load_state_dict(checkpoint["state_dict"])
best_epoch = int(checkpoint["best_epoch"])
best_val_auprc = float(checkpoint["validation_auprc"])

# ==================================================
# VALIDATION THRESHOLD
# ==================================================

val_labels, val_scores = get_scores(val_loader)

precision, recall, thresholds = precision_recall_curve(
    val_labels, val_scores
)

f1_values = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-12)
)

best_index = int(np.argmax(f1_values))

best_threshold = float(thresholds[best_index])


# ==================================================
# FINAL TEST EVALUATION
# ==================================================

test_labels, test_scores = get_scores(test_loader)

test_predictions = (
    test_scores >= best_threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    test_labels,
    test_predictions,
    labels=[0, 1]
).ravel()

results = {
    "Model": "Temporal Attention LSTM",
    "Best Epoch": best_epoch,
    "Validation AUPRC": float(best_val_auprc),
    "Threshold": best_threshold,
    "AUROC": float(roc_auc_score(test_labels, test_scores)),
    "AUPRC": float(average_precision_score(test_labels, test_scores)),
    "Precision": float(
        precision_score(test_labels, test_predictions, zero_division=0)
    ),
    "Recall": float(
        recall_score(test_labels, test_predictions, zero_division=0)
    ),
    "F1": float(
        f1_score(test_labels, test_predictions, zero_division=0)
    ),
    "TN": int(tn),
    "FP": int(fp),
    "FN": int(fn),
    "TP": int(tp),
    "Train Sequences": int(len(y_train)),
    "Validation Sequences": int(len(y_val)),
    "Test Sequences": int(len(y_test))
}


# ==================================================
# SAVE RESULTS AND CONFIGURATION
# ==================================================

pd.DataFrame(history).to_csv(
    RESULT_DIR / "lstm_training_history.csv",
    index=False
)

with open(
    RESULT_DIR / "lstm_test_results.json",
    "w"
) as f:
    json.dump(results, f, indent=4)

joblib.dump(
    {
        "features": FEATURES,
        "sequence_length": SEQUENCE_LENGTH,
        "hidden_size": HIDDEN_SIZE,
        "threshold": best_threshold
    },
    MODEL_DIR / "six_model_lstm_config.joblib"
)

print("\n" + "=" * 60)
print("TEMPORAL ATTENTION LSTM - FINAL TEST RESULTS")
print("=" * 60)

for key, value in results.items():
    print(f"{key}: {value}")

print("\nBest checkpoint:", checkpoint_path)
print("Results:", RESULT_DIR / "lstm_test_results.json")
print("\nLSTM training completed successfully.")
