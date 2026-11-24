from pathlib import Path
import streamlit as st
import pandas as pd

from utils.research_model_loader import load_research_models
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
# LOAD MODELS (CACHED)
# -------------------------------------------------
@st.cache_resource(show_spinner="Loading research models...")
def get_research_models():
    return load_research_models()


@st.cache_resource(show_spinner="Loading legacy model configuration...")
def get_legacy_models():
    return load_models()


try:
    research_models = get_research_models()
    st.session_state["research_models"] = research_models
except Exception as error:
    st.error(f"Unable to load six-model research suite: {error}")
    st.stop()

try:
    legacy_models = get_legacy_models()
    st.session_state["models"] = legacy_models
except Exception:
    # Legacy models optional fallback if research suite is primary
    pass


# -------------------------------------------------
# SIDEBAR NAVIGATION
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
    st.caption("Six-Model Research Suite | Local Environment")


# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("Clinical Deterioration Detection")

st.subheader(
    "Explainable Temporal Machine Learning Using Vital Signs"
)

st.write(
    "A clinical decision-support and research prototype exploring how machine learning "
    "and deep learning models identify patterns associated with early clinical "
    "deterioration in hospitalized and ICU patients using hourly vital-sign trajectories."
)


# -------------------------------------------------
# SIX-MODEL AVAILABILITY
# -------------------------------------------------
st.divider()

st.subheader("Model Availability")
st.write(
    "The application integrates six research models trained on temporal and clinical vital signs:"
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Original XGBoost", "Loaded", "48 features")
    st.metric("Linear SVM", "Loaded", "48 features")

with col2:
    st.metric("Random Forest", "Loaded", "300 trees | 48 feat")
    st.metric("Clinical-only XGBoost", "Loaded", "46 features")

with col3:
    st.metric("Logistic Regression", "Loaded", "Standardized | 48 feat")
    st.metric("Temporal Attention LSTM", "Loaded", "14 feat × 6 hours")


# -------------------------------------------------
# RESEARCH DATASET HIGHLIGHTS
# -------------------------------------------------
st.divider()

st.subheader("Research Dataset")

st.write(
    "The research models were trained and evaluated on an experimental subset of the "
    "**PhysioNet Computing in Cardiology Challenge 2019** sepsis prediction dataset."
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Patient Files", "11,472")

with col2:
    st.metric("Hourly Records", "445,177")

with col3:
    st.metric("Positive Sepsis Cases", "986")

with col4:
    st.metric("Common Test Cohort", "58,132 obs", "1,721 patients | 1,300 pos")


# -------------------------------------------------
# RESEARCH MODEL PERFORMANCE (COMMON TEST BENCHMARK)
# -------------------------------------------------
st.divider()

st.subheader("Research Model Benchmark Performance")

st.write(
    "Rigorous comparative evaluation on the common held-out test cohort "
    "(58,132 eligible hourly endpoints, 1,721 patients, 1,300 positive labels, ~2.24% positive class)."
)

# Load common test comparison dynamically
common_comp_path = (
    BASE_DIR
    / "experiments"
    / "six_model_comparison"
    / "results"
    / "common_test_model_comparison.csv"
)

if common_comp_path.exists():
    try:
        benchmark_df = pd.read_csv(common_comp_path)
        display_df = benchmark_df[
            ["Model", "AUROC", "AUPRC", "Precision", "Recall", "F1", "Threshold"]
        ].copy()

        st.dataframe(
            display_df.style.format({
                "AUROC": "{:.4f}",
                "AUPRC": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}",
                "Threshold": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )
    except Exception as e:
        st.warning(f"Could not load benchmark table dynamically: {e}")
else:
    st.info("Benchmark performance CSV not found at default path.")

# Progress bars for AUROC
st.markdown("#### AUROC Summary")
benchmark_summary = [
    ("Original XGBoost", 0.7912),
    ("Linear SVM", 0.7793),
    ("Logistic Regression", 0.7783),
    ("Random Forest", 0.7699),
    ("Clinical-only XGBoost", 0.7055),
    ("Temporal Attention LSTM", 0.6861)
]

for model_name, auroc in benchmark_summary:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(model_name)
        st.progress(auroc)
    with col2:
        st.write(f"**{auroc:.4f} AUROC**")

st.caption(
    "Evaluation metrics are experimental research results on a held-out test split, "
    "not prospective measures of clinical effectiveness."
)


# -------------------------------------------------
# PROJECT OVERVIEW CARDS
# -------------------------------------------------
st.divider()

st.subheader("Explore the Platform")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Six-Model Prediction")
    st.write(
        "Evaluate consecutive patient vital-sign observations across six machine learning "
        "and deep learning architectures, comparing alert thresholds and risk estimates."
    )

    st.markdown("### Temporal & Physiological Dynamics")
    st.write(
        "Analyze rolling vital trajectories (3h and 6h windows), rate-of-change deltas, "
        "Shock Index, and Modified Shock Index."
    )

with col2:
    st.markdown("### Multi-Angle Explainability")
    st.write(
        "Explore individual patient SHAP feature contributions, global beeswarm summaries, "
        "built-in tree importance, and recurrent LSTM temporal attention distributions."
    )

    st.markdown("### Comprehensive Research Dashboard")
    st.write(
        "Review feature ablation experiments, paired bootstrap statistical tests, "
        "lead-time analysis, continuous early warnings, and alert episode distributions."
    )


# -------------------------------------------------
# START ASSESSMENT
# -------------------------------------------------
st.divider()

st.subheader("Start a New Assessment")

st.write(
    "Enter patient observations manually, upload a multi-hour observation CSV, "
    "or explore synthetic demo scenarios."
)

if st.button("Start New Assessment", type="primary"):
    st.switch_page("pages/2_New_Assessment.py")


# -------------------------------------------------
# DISCLAIMER
# -------------------------------------------------
st.divider()

st.info(
    "**Academic Research Prototype Disclaimer:**\n\n"
    "This system is developed strictly for research, benchmarking, and educational purposes. "
    "The models have not undergone prospective clinical trials, clearance, or validation as a medical device. "
    "The outputs are decision-support estimates and must not replace professional medical judgment, "
    "clinical diagnosis, or patient management protocols."
)
