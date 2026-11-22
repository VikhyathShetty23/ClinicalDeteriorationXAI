
from utils.model_loader import load_models
from utils.preprocessing import prepare_model_inputs
from utils.prediction import predict_all

import pandas as pd


# Load trained models
models = load_models()

# Create six sample hourly observations
sample_data = pd.DataFrame({
    "HR": [82, 85, 88, 92, 96, 100],
    "O2Sat": [98, 98, 97, 96, 95, 94],
    "Temp": [37.0, 37.1, 37.2, 37.4, 37.5, 37.7],
    "SBP": [120, 118, 116, 112, 108, 104],
    "MAP": [85, 84, 82, 80, 77, 74],
    "Resp": [18, 19, 20, 21, 23, 24],
    "Age": [65] * 6,
    "Gender": [1] * 6,
    "ICULOS": [10, 11, 12, 13, 14, 15],
    "HospAdmTime": [-5] * 6
})

# Prepare model inputs
model_inputs = prepare_model_inputs(
    sample_data,
    models["config"],
    models["xgb_imputer"],
    models["clinical_imputer"],
    models["lstm_imputer"],
    models["lstm_scaler"]
)

# Generate predictions
results = predict_all(models, model_inputs)

# Display results
print("\nPREDICTION RESULTS")
print("-" * 40)

for name, probability in results["probabilities"].items():

    threshold = results["thresholds"][name]
    alert = results["alerts"][name]

    print(f"\nModel: {name}")
    print(f"Probability: {probability:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Model alert: {'YES' if alert else 'NO'}")

print("\nLSTM Attention Weights:")
print(results["lstm_attention"])
