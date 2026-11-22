
import streamlit as st
from pathlib import Path

from utils.model_loader import load_models


# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CSS_PATH = BASE_DIR / "assets" / "style.css"

st.set_page_config(
    page_title="Clinical Deterioration XAI",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -------------------------------------------------
# CUSTOM STYLING
# -------------------------------------------------
if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as file:
        st.markdown(
            f"<style>{file.read()}</style>",
            unsafe_allow_html=True
        )


# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
@st.cache_resource
def get_models():
    return load_models()


try:
    models = get_models()
    st.session_state["models"] = models
except Exception as error:
    st.error(f"Unable to load models: {error}")
    st.stop()


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
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


# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("Clinical Deterioration Detection")

st.subheader(
    "Explainable Temporal Machine Learning Using Vital Signs"
)

st.write(
    "A research prototype exploring how machine learning models "
    "can identify patterns associated with early clinical "
    "deterioration using hourly patient observations."
)


# -------------------------------------------------
# MODEL AVAILABILITY
# -------------------------------------------------
st.divider()

st.subheader("Model Availability")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Original XGBoost", "Loaded")

with col2:
    st.metric("Clinical-only XGBoost", "Loaded")

with col3:
    st.metric("Temporal Attention LSTM", "Loaded")


# -------------------------------------------------
# RESEARCH DATASET HIGHLIGHTS
# -------------------------------------------------
st.divider()

st.subheader("Research Dataset")

st.write(
    "The experiments use an available subset of the "
    "PhysioNet 2019 sepsis prediction dataset."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Patient files",
        "11,472"
    )

with col2:
    st.metric(
        "Hourly records",
        "445,177"
    )

with col3:
    st.metric(
        "Patients with positive labels",
        "986"
    )


# -------------------------------------------------
# MODEL PERFORMANCE
# -------------------------------------------------
st.divider()

st.subheader("Research Model Performance")

st.write(
    "Held-out endpoint test evaluation results."
)

performance = [
    ("Original XGBoost", 0.7767),
    ("Clinical-only XGBoost", 0.6759),
    ("Improved LSTM", 0.6968)
]

for model_name, auroc in performance:

    col1, col2 = st.columns([3, 1])

    with col1:
        st.write(model_name)
        st.progress(auroc)

    with col2:
        st.write(f"**{auroc:.4f} AUROC**")


st.caption(
    "AUROC values are experimental evaluation results, "
    "not measures of clinical effectiveness."
)


# -------------------------------------------------
# PROJECT OVERVIEW
# -------------------------------------------------
st.divider()

st.subheader("Explore the Project")

col1, col2 = st.columns(2)

with col1:

    st.markdown("### Prediction")

    st.write(
        "Compare predictions from three trained machine learning "
        "models using patient vital-sign observations."
    )

    st.markdown("### Temporal Analysis")

    st.write(
        "Examine changes in vital signs across consecutive "
        "hourly observations."
    )

with col2:

    st.markdown("### Explainability")

    st.write(
        "Explore SHAP-based XGBoost feature contributions and "
        "temporal attention weights from the LSTM."
    )

    st.markdown("### Research Evaluation")

    st.write(
        "Review model performance, early-warning analysis, "
        "and research limitations."
    )


# -------------------------------------------------
# START ASSESSMENT
# -------------------------------------------------
st.divider()

st.subheader("Start a New Assessment")

st.write(
    "Enter patient observations manually or upload a CSV file "
    "to explore predictions from the trained models."
)

if st.button("Start New Assessment", type="primary"):

    st.switch_page("pages/2_New_Assessment.py")


# -------------------------------------------------
# DISCLAIMER
# -------------------------------------------------
st.divider()

st.caption(
    "Research prototype for educational purposes only. "
    "The models have not been clinically validated and must "
    "not be used for diagnosis or treatment decisions."
)
