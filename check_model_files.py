
import joblib
import torch
from pathlib import Path

model_dir = Path("models")

files = [
    "model_config.joblib",
    "xgboost_original.joblib",
    "xgboost_clinical_only.joblib",
    "xgboost_imputer.joblib",
    "clinical_imputer.joblib",
    "lstm_imputer.joblib",
    "lstm_scaler.joblib",
]

print("Checking JOBLIB files...\n")

for filename in files:
    try:
        obj = joblib.load(model_dir / filename)
        print(f"OK: {filename}")
    except Exception as e:
        print(f"FAILED: {filename}")
        print("Error:", repr(e))

print("\nChecking LSTM checkpoint...\n")

try:
    state = torch.load(
        model_dir / "improved_lstm.pth",
        map_location="cpu",
        weights_only=True
    )
    print("OK: improved_lstm.pth")
except Exception as e:
    print("FAILED: improved_lstm.pth")
    print("Error:", repr(e))
