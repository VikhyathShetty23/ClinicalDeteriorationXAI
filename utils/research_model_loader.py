from pathlib import Path
import joblib
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_MODEL_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "six_model_comparison"
    / "saved_models"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TemporalAttentionLSTM(nn.Module):
    """
    Temporal Attention LSTM for sequence classification.
    Matches the exact research architecture trained in train_lstm.py.
    """

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
        attention_weights = torch.softmax(attention_scores, dim=1)

        context = torch.sum(attention_weights * lstm_output, dim=1)
        output = self.classifier(context)

        return output.squeeze(1), attention_weights.squeeze(-1)


def load_research_models():
    """
    Load all six research models and their corresponding preprocessing objects.

    Returns:
        dict containing:
            - tabular: dict of 5 tabular model artifacts (model, imputer, scaler, feature_columns, threshold, model_name)
            - lstm: dict of LSTM model, imputer, scaler, config, features, sequence_length, threshold
            - device: torch device used
    """
    required_files = [
        "original_xgboost.joblib",
        "clinical_only_xgboost.joblib",
        "random_forest.joblib",
        "logistic_regression.joblib",
        "linear_svm.joblib",
        "six_model_lstm_config.joblib",
        "six_model_lstm_imputer.joblib",
        "six_model_lstm_scaler.joblib",
        "six_model_temporal_attention_lstm.pth"
    ]

    missing = [
        fname for fname in required_files
        if not (RESEARCH_MODEL_DIR / fname).exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"Missing required research model artifacts in {RESEARCH_MODEL_DIR}: {missing}"
        )

    # 1. Load five tabular artifacts
    tabular_models = {
        "original_xgboost": joblib.load(RESEARCH_MODEL_DIR / "original_xgboost.joblib"),
        "clinical_only_xgboost": joblib.load(RESEARCH_MODEL_DIR / "clinical_only_xgboost.joblib"),
        "random_forest": joblib.load(RESEARCH_MODEL_DIR / "random_forest.joblib"),
        "logistic_regression": joblib.load(RESEARCH_MODEL_DIR / "logistic_regression.joblib"),
        "linear_svm": joblib.load(RESEARCH_MODEL_DIR / "linear_svm.joblib")
    }

    # Validate tabular artifact contents
    for key, artifact in tabular_models.items():
        if not isinstance(artifact, dict) or "model" not in artifact or "imputer" not in artifact:
            raise ValueError(f"Invalid artifact format for tabular model '{key}'. Expected dictionary with model and imputer.")

    # 2. Load LSTM artifacts
    lstm_config = joblib.load(RESEARCH_MODEL_DIR / "six_model_lstm_config.joblib")
    lstm_imputer = joblib.load(RESEARCH_MODEL_DIR / "six_model_lstm_imputer.joblib")
    lstm_scaler = joblib.load(RESEARCH_MODEL_DIR / "six_model_lstm_scaler.joblib")

    input_size = len(lstm_config["features"])
    hidden_size = lstm_config["hidden_size"]

    lstm_model = TemporalAttentionLSTM(
        input_size=input_size,
        hidden_size=hidden_size
    )

    checkpoint = torch.load(
        RESEARCH_MODEL_DIR / "six_model_temporal_attention_lstm.pth",
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    lstm_model.load_state_dict(state_dict, strict=True)
    lstm_model.to(DEVICE)
    lstm_model.eval()

    lstm_bundle = {
        "model": lstm_model,
        "imputer": lstm_imputer,
        "scaler": lstm_scaler,
        "config": lstm_config,
        "features": lstm_config["features"],
        "sequence_length": lstm_config["sequence_length"],
        "threshold": float(lstm_config["threshold"]),
        "model_name": "Temporal Attention LSTM"
    }

    return {
        "tabular": tabular_models,
        "lstm": lstm_bundle,
        "device": DEVICE
    }
