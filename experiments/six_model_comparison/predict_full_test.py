
from pathlib import Path
import joblib
import pandas as pd

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = Path(
    r"C:\Users\Vikhyath Shetty\Downloads\Sepsis_Dataset\prepared_features.csv"
)

MODEL_PATH = BASE_DIR / "saved_models" / "original_xgboost.joblib"

OUTPUT_DIR = BASE_DIR / "lead_time"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "original_xgboost_full_test_predictions.csv"


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("Loading saved Original XGBoost model...")

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
imputer = artifact["imputer"]
scaler = artifact["scaler"]
features = artifact["feature_columns"]

print("Model:", artifact["model_name"])
print("Features:", len(features))


# --------------------------------------------------
# LOAD TEST DATA
# --------------------------------------------------

print("\nLoading prepared dataset...")

df = pd.read_csv(DATA_PATH)

test_df = df[
    df["Split"].str.lower() == "test"
].copy()

print("Test records:", len(test_df))
print("Test patients:", test_df["PatientID"].nunique())


# --------------------------------------------------
# PREPROCESS USING SAVED OBJECTS
# --------------------------------------------------

print("\nApplying saved preprocessing...")

X_test = test_df[features]

X_test = imputer.transform(X_test)

if scaler is not None:
    X_test = scaler.transform(X_test)


# --------------------------------------------------
# GENERATE PREDICTIONS
# --------------------------------------------------

print("\nGenerating predictions...")

scores = model.predict_proba(X_test)[:, 1]

threshold = 0.748707

predictions = pd.DataFrame({
    "PatientID": test_df["PatientID"].astype(str).to_numpy(),
    "ICULOS": test_df["ICULOS"].to_numpy(),
    "SepsisLabel": test_df["SepsisLabel"].astype(int).to_numpy(),
    "original_xgboost_score": scores,
    "PredictedLabel": (scores >= threshold).astype(int)
})


# --------------------------------------------------
# SAVE
# --------------------------------------------------

predictions.to_csv(OUTPUT_PATH, index=False)

print("\nFull test predictions generated successfully!")
print("Prediction rows:", len(predictions))
print("Unique patients:", predictions["PatientID"].nunique())
print("Positive labels:", int(predictions["SepsisLabel"].sum()))
print("Predicted positive:", int(predictions["PredictedLabel"].sum()))

print("\nSaved to:")
print(OUTPUT_PATH)
