from typing import Dict, Any
import numpy as np
import pandas as pd
import torch

from utils.preprocessing import (
    validate_observations,
    handle_missing_values,
    create_temporal_features
)


def predict_six_models(
    research_models: Dict[str, Any],
    observations: pd.DataFrame
) -> Dict[str, Any]:
    """
    Generate predictions from all six research models.

    Args:
        research_models: Dictionary returned by load_research_models().
        observations: Raw DataFrame containing patient observations.

    Returns:
        Dictionary containing:
            - model_outputs: Detailed dictionary of all six model results
            - probabilities: Dictionary of probabilities for the 5 probabilistic models
            - decision_scores: Dictionary of decision scores for Linear SVM
            - thresholds: Dictionary of thresholds for all models
            - alerts: Dictionary of boolean alert flags
            - lstm_attention: List of 6 temporal attention weights
            - processed_data: Fully engineered feature DataFrame
            - prepared_inputs: Dictionary of preprocessed 2D arrays for tabular models
    """
    # 1. Validation & Preprocessing
    data = validate_observations(observations)
    data = handle_missing_values(data)
    data = create_temporal_features(data)

    # Replace infinite values with NaN before imputation
    data = data.replace([np.inf, -np.inf], np.nan)

    tabular_models = research_models["tabular"]
    device = research_models["device"]

    model_outputs = {}
    probabilities = {}
    decision_scores = {}
    thresholds = {}
    alerts = {}
    prepared_inputs = {}

    # 2. Predict Tabular Models
    tabular_order = [
        ("original_xgboost", "Original XGBoost"),
        ("random_forest", "Random Forest"),
        ("logistic_regression", "Logistic Regression"),
        ("linear_svm", "Linear SVM"),
        ("clinical_only_xgboost", "Clinical Only XGBoost")
    ]

    for key, display_name in tabular_order:
        artifact = tabular_models[key]
        model = artifact["model"]
        imputer = artifact["imputer"]
        scaler = artifact["scaler"]
        cols = artifact["feature_columns"]
        threshold = float(artifact["threshold"])

        # Extract features in exact training order
        X = data[cols].copy()
        X = imputer.transform(X)
        if scaler is not None:
            X = scaler.transform(X)

        # Use latest observation for prediction
        X_latest = X[-1:].astype(np.float32)
        prepared_inputs[key] = X_latest

        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X_latest)[0, 1])
            alert = bool(prob >= threshold)
            output_type = "probability"
            score = prob
            probabilities[key] = prob
        else:
            # Linear SVM uses decision_function
            dec_score = float(model.decision_function(X_latest)[0])
            alert = bool(dec_score >= threshold)
            output_type = "decision_score"
            score = dec_score
            decision_scores[key] = dec_score

        thresholds[key] = threshold
        alerts[key] = alert

        model_outputs[key] = {
            "name": display_name,
            "key": key,
            "model_type": "tabular",
            "output_type": output_type,
            "score": score,
            "threshold": threshold,
            "alert": alert,
            "feature_columns": cols
        }

    # 3. Predict Temporal Attention LSTM
    lstm_bundle = research_models["lstm"]
    lstm_model = lstm_bundle["model"]
    lstm_imputer = lstm_bundle["imputer"]
    lstm_scaler = lstm_bundle["scaler"]
    lstm_features = lstm_bundle["features"]
    seq_len = lstm_bundle["sequence_length"]
    lstm_thresh = float(lstm_bundle["threshold"])

    if len(data) < seq_len:
        raise ValueError(
            f"Temporal Attention LSTM requires at least {seq_len} consecutive observations."
        )

    # Prepare sequence
    seq_data = data[lstm_features].to_numpy(dtype=np.float32)
    seq_data = lstm_imputer.transform(seq_data)
    seq_data = lstm_scaler.transform(seq_data)
    seq_data = seq_data[-seq_len:][np.newaxis, :, :]  # (1, 6, 14)

    lstm_model.eval()
    with torch.no_grad():
        seq_tensor = torch.tensor(seq_data, dtype=torch.float32, device=device)
        logits, attention_weights = lstm_model(seq_tensor)
        lstm_prob = float(torch.sigmoid(logits).item())
        attention_list = attention_weights.squeeze(0).cpu().numpy().tolist()

    lstm_alert = bool(lstm_prob >= lstm_thresh)
    probabilities["temporal_attention_lstm"] = lstm_prob
    thresholds["temporal_attention_lstm"] = lstm_thresh
    alerts["temporal_attention_lstm"] = lstm_alert

    model_outputs["temporal_attention_lstm"] = {
        "name": "Temporal Attention LSTM",
        "key": "temporal_attention_lstm",
        "model_type": "recurrent_lstm",
        "output_type": "probability",
        "score": lstm_prob,
        "threshold": lstm_thresh,
        "alert": lstm_alert,
        "attention_weights": attention_list
    }

    return {
        "model_outputs": model_outputs,
        "probabilities": probabilities,
        "decision_scores": decision_scores,
        "thresholds": thresholds,
        "alerts": alerts,
        "lstm_attention": attention_list,
        "processed_data": data,
        "prepared_inputs": prepared_inputs
    }
