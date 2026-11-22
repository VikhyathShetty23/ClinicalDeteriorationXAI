
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from utils.preprocessing import prepare_model_inputs
from utils.explainability import (
    explain_xgboost,
    get_model_feature_importance
)


st.set_page_config(
    page_title="Explainability | Clinical XAI",
    layout="wide"
)

# Shared styling
BASE_DIR = Path(__file__).resolve().parents[1]
CSS_PATH = BASE_DIR / "assets" / "style.css"

if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as file:
        st.markdown(
            f"<style>{file.read()}</style>",
            unsafe_allow_html=True
        )


# Sidebar
with st.sidebar:
    st.markdown("## Clinical XAI")
    st.caption("Early Deterioration Detection")
    st.divider()

    st.markdown("**Navigation**")
    st.page_link("app.py", label="Overview")
    st.page_link("pages/2_New_Assessment.py", label="New Assessment")
    st.page_link("pages/3_Prediction_Results.py", label="Prediction Results")
    st.page_link("pages/4_Explainability.py", label="Explainability")
    st.page_link("pages/5_Trends_Research.py", label="Trends & Research")

    st.divider()
    st.caption("Research Prototype | Local Environment")


# Page heading
st.title("Model Explainability")

st.write(
    "Explore feature contributions from XGBoost and temporal "
    "attention weights from the LSTM."
)


# Retrieve assessment
models = st.session_state.get("models")
observations = st.session_state.get("assessment_observations")
results = st.session_state.get("assessment_results")

if models is None or observations is None or results is None:
    st.info(
        "Complete a new assessment first to view its explanations."
    )
    st.page_link(
        "pages/2_New_Assessment.py",
        label="Go to New Assessment"
    )
    st.stop()


# Recreate inputs using the same preprocessing pipeline
try:
    model_inputs = prepare_model_inputs(
        observations,
        models["config"],
        models["xgb_imputer"],
        models["clinical_imputer"],
        models["lstm_imputer"],
        models["lstm_scaler"]
    )
except Exception as error:
    st.error(f"Unable to prepare explanation inputs: {error}")
    st.stop()


# =====================================================
# XGBOOST SHAP EXPLANATION
# =====================================================
st.divider()
st.subheader("XGBoost Feature Contributions")

model_choice = st.selectbox(
    "Choose XGBoost model",
    ["Original XGBoost", "Clinical-only XGBoost"]
)

if model_choice == "Original XGBoost":
    selected_model = models["original_xgb"]
    selected_input = model_inputs["original_xgb"]
    feature_names = models["config"]["xgb_features"]
else:
    selected_model = models["clinical_xgb"]
    selected_input = model_inputs["clinical_xgb"]
    feature_names = models["config"]["clinical_features"]


from utils.explainability import explain_xgboost

try:
    explanation = explain_xgboost(
        selected_model,
        selected_input,
        feature_names
    )

    contributions = explanation["contributions"]

    st.write(
        "The chart shows the strongest feature contributions "
        "for the selected assessment."
    )

    top_contributions = contributions[:15]

    chart_data = pd.DataFrame(top_contributions)

    fig, ax = plt.subplots(figsize=(10, 6))

    values = chart_data["value"].to_numpy()
    labels = chart_data["feature"].to_numpy()

    bar_colors = [
        "#367f7a" if value >= 0 else "#c47b55"
        for value in values
    ]

    ax.barh(labels[::-1], values[::-1], color=bar_colors[::-1])

    ax.axvline(0, color="#263445", linewidth=0.8)

    ax.set_xlabel("SHAP contribution (model output scale)")
    ax.set_ylabel("Feature")
    ax.set_title("Top 15 Individual SHAP Contributions")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Positive SHAP values push the model output higher; "
        "negative values push it lower. These contributions are "
        "on the model output scale, not percentage-point changes "
        "in predicted probability."
    )

    with st.expander("View all SHAP contributions"):
        st.dataframe(
            pd.DataFrame(contributions),
            use_container_width=True,
            hide_index=True
        )

except Exception as error:
    st.error(f"Unable to generate SHAP explanation: {error}")


# =====================================================
# BUILT-IN XGBOOST FEATURE IMPORTANCE
# =====================================================
st.divider()
st.subheader("Trained Model Feature Importance")

st.write(
    "This chart uses the XGBoost model's built-in feature "
    "importance values. It is a model-level summary and is "
    "different from SHAP global importance."
)

try:
    importance = get_model_feature_importance(
        selected_model,
        feature_names
    )

    importance_df = pd.DataFrame(importance).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        importance_df["feature"].to_numpy()[::-1],
        importance_df["importance"].to_numpy()[::-1],
        color="#367f7a"
    )

    ax.set_xlabel("Feature importance")
    ax.set_ylabel("Feature")
    ax.set_title("Top 15 Built-in Feature Importance Values")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

except Exception as error:
    st.error(f"Unable to display feature importance: {error}")


# =====================================================
# LSTM TEMPORAL ATTENTION
# =====================================================
st.divider()
st.subheader("LSTM Temporal Attention")

attention = results.get("lstm_attention", [])

if len(attention) == 6:

    hours = observations["ICULOS"].astype(int).tolist()

    attention_df = pd.DataFrame({
        "ICU Hour": [f"Hour {hour}" for hour in hours],
        "Attention Weight": attention
    })

    st.write(
        "The LSTM assigns an attention weight to each of the "
        "six consecutive hourly observations."
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(
        attention_df["ICU Hour"],
        attention_df["Attention Weight"],
        color="#367f7a"
    )

    ax.set_ylabel("Attention weight")
    ax.set_xlabel("ICU observation hour")
    ax.set_title("Temporal Attention Distribution")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(
        attention_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Attention weights sum to approximately 1. They describe "
        "how the model distributes attention across time steps; "
        "they do not establish causality or clinical importance."
    )

else:
    st.warning(
        "LSTM attention weights are unavailable for this assessment."
    )


# =====================================================
# LIMITATIONS
# =====================================================
st.divider()
st.subheader("Interpretation and Limitations")

st.info(
    """
    SHAP contributions explain how input features influence the
    selected XGBoost model output for this assessment. They do not
    establish causal relationships.

    Built-in feature importance is a separate model-level measure.

    LSTM attention weights describe the model's attention across
    the six input observations. They should not be interpreted as
    causal explanations.

    All explanations are intended for research and educational
    purposes, not clinical decision-making.
    """
)
