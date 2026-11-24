import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure root is in path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from utils.research_model_loader import load_research_models
from utils.research_prediction import predict_six_models
from utils.explainability import explain_xgboost, get_model_feature_importance


def test_six_models_pipeline():
    print("=" * 60)
    print("RUNNING SIX-MODEL INTEGRATION TEST")
    print("=" * 60)

    # 1. Test Model Loading
    print("\n[Step 1] Loading six-model research suite...")
    research_models = load_research_models()
    print("SUCCESS: Loaded all models and preprocessing objects.")
    print("Tabular models:", list(research_models["tabular"].keys()))
    print("LSTM loaded on device:", research_models["device"])

    # 2. Test Input Data
    print("\n[Step 2] Preparing sample 6-hour patient trajectory...")
    sample_df = pd.DataFrame({
        "ICULOS": [10, 11, 12, 13, 14, 15],
        "HR": [82.0, 85.0, 88.0, 92.0, 96.0, 100.0],
        "O2Sat": [98.0, 98.0, 97.0, 96.0, 95.0, 94.0],
        "Temp": [37.0, 37.1, 37.2, 37.4, 37.5, 37.7],
        "SBP": [120.0, 118.0, 116.0, 112.0, 108.0, 104.0],
        "MAP": [85.0, 84.0, 82.0, 80.0, 77.0, 74.0],
        "Resp": [18.0, 19.0, 20.0, 21.0, 23.0, 24.0],
        "Age": [65] * 6,
        "Gender": [1] * 6,
        "HospAdmTime": [-5.0] * 6
    })

    # 3. Test Prediction
    print("\n[Step 3] Executing six-model prediction...")
    results = predict_six_models(research_models, sample_df)
    outputs = results["model_outputs"]

    for key, out in outputs.items():
        if out["output_type"] == "probability":
            print(f"  - {out['name']}: Probability = {out['score'] * 100:.2f}%, Threshold = {out['threshold'] * 100:.2f}%, Alert = {out['alert']}")
        else:
            print(f"  - {out['name']}: Decision Score = {out['score']:+.4f}, Threshold = {out['threshold']:+.4f}, Alert = {out['alert']}")

    # 4. Assertions
    assert "linear_svm" in outputs, "Linear SVM missing from outputs"
    assert outputs["linear_svm"]["output_type"] == "decision_score", "Linear SVM must output decision_score"
    assert "temporal_attention_lstm" in outputs, "LSTM missing from outputs"
    assert len(results["lstm_attention"]) == 6, f"Expected 6 attention weights, got {len(results['lstm_attention'])}"
    assert np.isclose(sum(results["lstm_attention"]), 1.0, atol=1e-3), "LSTM attention weights must sum to approximately 1.0"

    # 5. Test SHAP Explanations
    print("\n[Step 4] Testing SHAP on Original XGBoost and Clinical-only XGBoost...")
    orig_art = research_models["tabular"]["original_xgboost"]
    orig_inp = results["prepared_inputs"]["original_xgboost"]
    orig_cols = orig_art["feature_columns"]
    exp_orig = explain_xgboost(orig_art["model"], orig_inp, orig_cols)
    assert len(exp_orig["contributions"]) == 48, f"Expected 48 SHAP contributions, got {len(exp_orig['contributions'])}"
    print(f"  - Original XGBoost SHAP: {len(exp_orig['contributions'])} features computed successfully.")

    clin_art = research_models["tabular"]["clinical_only_xgboost"]
    clin_inp = results["prepared_inputs"]["clinical_only_xgboost"]
    clin_cols = clin_art["feature_columns"]
    exp_clin = explain_xgboost(clin_art["model"], clin_inp, clin_cols)
    assert len(exp_clin["contributions"]) == 46, f"Expected 46 SHAP contributions, got {len(exp_clin['contributions'])}"
    print(f"  - Clinical-only XGBoost SHAP: {len(exp_clin['contributions'])} features computed successfully.")

    # 6. Test Built-in Feature Importance
    imp = get_model_feature_importance(orig_art["model"], orig_cols)
    assert len(imp) == 48, f"Expected 48 importance values, got {len(imp)}"
    print(f"  - Built-in feature importance: {len(imp)} features computed successfully.")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_six_models_pipeline()
