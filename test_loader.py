
from utils.model_loader import load_models

try:
    models = load_models()

    print("\nAll models loaded successfully!")

    print("Original XGBoost:", type(models["original_xgb"]))
    print("Clinical XGBoost:", type(models["clinical_xgb"]))
    print("LSTM:", type(models["lstm"]))

    print("LSTM device:", models["device"])
    print("LSTM hidden size:", models["config"]["lstm_hidden_size"])

except Exception as e:
    print("Model loading failed:")
    print(e)
