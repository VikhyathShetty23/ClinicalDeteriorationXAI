
from pathlib import Path

import joblib
import torch
import torch.nn as nn
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def load_models():

    required_files = [
        "model_config.joblib",
        "xgboost_original.json",
        "xgboost_clinical_only.json",
        "improved_lstm.pth",
        "xgboost_imputer.joblib",
        "clinical_imputer.joblib",
        "lstm_imputer.joblib",
        "lstm_scaler.joblib"
    ]

    missing_files = [
        name for name in required_files
        if not (MODEL_DIR / name).exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Missing model files: {missing_files}"
        )

    # Load model configuration
    config = joblib.load(
        MODEL_DIR / "model_config.joblib"
    )

    # Load original XGBoost model
    original_xgb = XGBClassifier()
    original_xgb.load_model(
        str(MODEL_DIR / "xgboost_original.json")
    )

    # Load clinical-only XGBoost model
    clinical_xgb = XGBClassifier()
    clinical_xgb.load_model(
        str(MODEL_DIR / "xgboost_clinical_only.json")
    )

    # Load preprocessing objects
    xgb_imputer = joblib.load(
        MODEL_DIR / "xgboost_imputer.joblib"
    )

    clinical_imputer = joblib.load(
        MODEL_DIR / "clinical_imputer.joblib"
    )

    lstm_imputer = joblib.load(
        MODEL_DIR / "lstm_imputer.joblib"
    )

    lstm_scaler = joblib.load(
        MODEL_DIR / "lstm_scaler.joblib"
    )

    # Reconstruct LSTM architecture
    lstm_model = TemporalAttentionLSTM(
        input_size=len(config["lstm_features"]),
        hidden_size=config["lstm_hidden_size"]
    )

    # Load trained LSTM weights
    checkpoint = torch.load(
        MODEL_DIR / "improved_lstm.pth",
        map_location=DEVICE,
        weights_only=True
    )

    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    lstm_model.load_state_dict(checkpoint, strict=True)

    lstm_model.to(DEVICE)
    lstm_model.eval()

    return {
        "config": config,
        "original_xgb": original_xgb,
        "clinical_xgb": clinical_xgb,
        "lstm": lstm_model,
        "xgb_imputer": xgb_imputer,
        "clinical_imputer": clinical_imputer,
        "lstm_imputer": lstm_imputer,
        "lstm_scaler": lstm_scaler,
        "device": DEVICE
    }
