
from pathlib import Path
import xgboost
import joblib

print("XGBoost version:", xgboost.__version__)
print("\nModel file sizes:")

for name in [
    "xgboost_original.joblib",
    "xgboost_clinical_only.joblib"
]:
    path = Path("models") / name
    print(f"{name}: {path.stat().st_size:,} bytes")

print("\nTesting XGBoost model loading:")

for name in [
    "xgboost_original.joblib",
    "xgboost_clinical_only.joblib"
]:
    try:
        model = joblib.load(Path("models") / name)
        print(f"SUCCESS: {name}")
    except Exception as e:
        print(f"FAILED: {name} - {repr(e)}")
