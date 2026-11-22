
import numpy as np
import shap


def explain_xgboost(model, model_input, feature_names):
    """
    Generate SHAP contributions for one XGBoost prediction.
    """

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(model_input)

    # Handle binary-class output
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[0, :, 1]
    else:
        shap_values = shap_values.reshape(-1)

    if len(shap_values) != len(feature_names):
        raise ValueError(
            "SHAP output does not match the feature count."
        )

    # Extract expected/base value
    base_value = explainer.expected_value

    if isinstance(base_value, (list, np.ndarray)):
        base_value = np.asarray(base_value).reshape(-1)[-1]

    contributions = [
        {
            "feature": name,
            "value": float(value)
        }
        for name, value in zip(feature_names, shap_values)
    ]

    contributions.sort(
        key=lambda item: abs(item["value"]),
        reverse=True
    )

    return {
        "base_value": float(base_value),
        "contributions": contributions
    }


def get_model_feature_importance(model, feature_names):
    """
    Return built-in XGBoost feature importance.

    This is not SHAP importance.
    """

    importance = model.feature_importances_

    if len(importance) != len(feature_names):
        raise ValueError(
            "Feature importance count does not match feature names."
        )

    result = [
        {
            "feature": name,
            "importance": float(value)
        }
        for name, value in zip(feature_names, importance)
    ]

    result.sort(
        key=lambda item: item["importance"],
        reverse=True
    )

    return result
