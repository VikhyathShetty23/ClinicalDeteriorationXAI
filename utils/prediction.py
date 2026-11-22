
import numpy as np
import torch


def predict_all(models, model_inputs):
    """
    Generate predictions from all three trained models.

    Args:
        models: Dictionary returned by load_models().
        model_inputs: Dictionary returned by prepare_model_inputs().

    Returns:
        Dictionary containing probabilities, alerts,
        thresholds, and LSTM attention weights.
    """

    config = models["config"]
    device = models["device"]

    # Retrieve saved thresholds
    thresholds = {
        "original_xgb": config["original_xgb_threshold"],
        "clinical_xgb": config["clinical_xgb_threshold"],
        "lstm": config["improved_lstm_threshold"]
    }

    # -----------------------------
    # 1. Original XGBoost prediction
    # -----------------------------
    original_probability = float(
        models["original_xgb"].predict_proba(
            model_inputs["original_xgb"]
        )[0][1]
    )

    # -----------------------------
    # 2. Clinical-only XGBoost
    # -----------------------------
    clinical_probability = float(
        models["clinical_xgb"].predict_proba(
            model_inputs["clinical_xgb"]
        )[0][1]
    )

    # -----------------------------
    # 3. Improved LSTM prediction
    # -----------------------------
    lstm_input = torch.tensor(
        model_inputs["lstm"],
        dtype=torch.float32,
        device=device
    )

    models["lstm"].eval()

    with torch.no_grad():
        logits, attention_weights = models["lstm"](lstm_input)

        lstm_probability = torch.sigmoid(logits).item()

        attention = (
            attention_weights
            .squeeze(0)
            .cpu()
            .numpy()
            .tolist()
        )

    # -----------------------------
    # Calculate threshold alerts
    # -----------------------------
    probabilities = {
        "original_xgb": original_probability,
        "clinical_xgb": clinical_probability,
        "lstm": lstm_probability
    }

    alerts = {
        name: probability >= thresholds[name]
        for name, probability in probabilities.items()
    }

    # -----------------------------
    # Return results
    # -----------------------------
    results = {
        "probabilities": probabilities,
        "thresholds": thresholds,
        "alerts": alerts,
        "lstm_attention": attention
    }

    return results
