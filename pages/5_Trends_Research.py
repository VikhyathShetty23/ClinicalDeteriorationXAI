
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


st.set_page_config(
    page_title="Trends & Research | Clinical XAI",
    layout="wide"
)

# -------------------------------------------------
# Shared styling
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
CSS_PATH = BASE_DIR / "assets" / "style.css"

if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as file:
        st.markdown(
            f"<style>{file.read()}</style>",
            unsafe_allow_html=True
        )


# -------------------------------------------------
# Sidebar
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
# Page heading
# -------------------------------------------------
st.title("Trends & Research")

st.write(
    "Explore patient vital-sign patterns and review the "
    "experimental performance of the trained models."
)


# =================================================
# SECTION 1: VITAL SIGN TRENDS
# =================================================
st.divider()
st.subheader("Patient Vital-Sign Trends")

observations = st.session_state.get("assessment_observations")

if observations is not None:

    vital_features = [
        "HR", "O2Sat", "Temp", "SBP", "MAP", "Resp"
    ]

    available_features = [
        col for col in vital_features
        if col in observations.columns
    ]

    selected_vital = st.selectbox(
        "Select vital sign",
        available_features
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        observations["ICULOS"],
        observations[selected_vital],
        marker="o",
        linewidth=2,
        color="#367f7a"
    )

    ax.set_xlabel("ICU observation hour")
    ax.set_ylabel(selected_vital)
    ax.set_title(f"{selected_vital} over time")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "The chart displays the submitted observations. Missing "
        "measurements are not presented as actual measured values."
    )

    with st.expander("View all vital-sign observations"):
        st.dataframe(
            observations[
                ["ICULOS"] + available_features
            ],
            use_container_width=True,
            hide_index=True
        )

else:
    st.info(
        "Complete an assessment to view its vital-sign trends."
    )


# =================================================
# SECTION 2: MODEL PERFORMANCE
# =================================================
st.divider()
st.subheader("Research Model Performance")

st.write(
    "The following values are from the held-out endpoint test "
    "evaluation of the research experiment, not from the latest "
    "synthetic assessment."
)

performance_data = pd.DataFrame({
    "Model": [
        "Original XGBoost",
        "Clinical-only XGBoost",
        "Improved LSTM"
    ],
    "AUROC": [0.7767, 0.6759, 0.6968],
    "AUPRC": [0.1016, 0.0573, 0.0563],
    "Precision": [0.1201, 0.0750, 0.0967],
    "Recall": [0.3748, 0.2427, 0.1553],
    "F1-score": [0.1819, 0.1146, 0.1192]
})

st.dataframe(
    performance_data.style.format({
        "AUROC": "{:.4f}",
        "AUPRC": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1-score": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)


# AUROC comparison
st.markdown("#### AUROC Comparison")

fig, ax = plt.subplots(figsize=(9, 4))

ax.bar(
    performance_data["Model"],
    performance_data["AUROC"],
    color=["#367f7a", "#8297a8", "#a6b8b5"]
)

ax.set_ylabel("AUROC")
ax.set_ylim(0, 1)
ax.set_title("Area Under the ROC Curve")
ax.tick_params(axis="x", rotation=10)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


# AUPRC comparison
st.markdown("#### AUPRC Comparison")

fig, ax = plt.subplots(figsize=(9, 4))

ax.bar(
    performance_data["Model"],
    performance_data["AUPRC"],
    color=["#367f7a", "#8297a8", "#a6b8b5"]
)

ax.set_ylabel("AUPRC")
ax.set_ylim(0, max(performance_data["AUPRC"]) * 1.3)
ax.set_title("Area Under the Precision-Recall Curve")
ax.tick_params(axis="x", rotation=10)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


st.caption(
    "The dataset is imbalanced. AUPRC is included alongside AUROC "
    "to describe performance on the positive class. These are "
    "experimental results, not evidence of clinical effectiveness."
)


# =================================================
# SECTION 3: EARLY WARNING ANALYSIS
# =================================================
st.divider()
st.subheader("Early-Warning Analysis")

st.write(
    "Exploratory analysis of the original XGBoost model's alert "
    "timing relative to the estimated sepsis onset."
)

col1, col2, col3 = st.columns(3)

col1.metric("Pre-onset alerts in analysis window", "62")
col2.metric("Alerts at or after estimated onset", "6")
col3.metric("Median useful lead time", "12 hours")

st.markdown("#### Alert timing summary")

early_warning_data = pd.DataFrame({
    "Category": [
        "Pre-onset alerts in analysis window",
        "Alerts at or after estimated onset",
        "Too early only",
        "No alert"
    ],
    "Patients": [62, 6, 5, 75]
})

fig, ax = plt.subplots(figsize=(9, 4))

ax.barh(
    early_warning_data["Category"][::-1],
    early_warning_data["Patients"][::-1],
    color="#367f7a"
)

ax.set_xlabel("Number of patients")
ax.set_title("Exploratory Alert Timing Distribution")

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.write(
    "**Estimated onset:** first positive SepsisLabel observation "
    "plus approximately six hours."
)

st.write(
    "**Analysis window:** 12 hours before to 3 hours after the "
    "estimated onset."
)

st.caption(
    "The onset estimate is approximate. The 62 pre-onset detections "
    "are not a claim of clinical sensitivity or prospective performance."
)


# =================================================
# SECTION 4: DATASET AND RESEARCH LIMITATIONS
# =================================================
st.divider()
st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Patient files", "11,472")
col2.metric("Hourly records", "445,177")
col3.metric("Patients with positive sepsis labels", "986")

st.write(
    "The project uses an available subset of the PhysioNet 2019 "
    "sepsis prediction dataset, not the complete official training cohort."
)


st.divider()
st.subheader("Research Limitations")

limitations = [
    (
        "Dataset scope",
        "The experiments use an available dataset subset and may not "
        "represent the full patient population."
    ),
    (
        "Class imbalance",
        "Positive hourly sepsis labels represent approximately 2.12% "
        "of the hourly records."
    ),
    (
        "Timing variables",
        "ICU timing variables materially influence the original "
        "XGBoost model. The clinical-only model's lower performance "
        "highlights the need to investigate this dependence."
    ),
    (
        "Explainability",
        "SHAP contributions and LSTM attention weights describe model "
        "behaviour but do not establish causality."
    ),
    (
        "NEWS2 comparison",
        "The NEWS2 comparison is partial and exploratory, not a full "
        "NEWS2 implementation."
    ),
    (
        "Clinical validation",
        "The models have not been prospectively or clinically validated "
        "for diagnosis, treatment, or escalation decisions."
    )
]

for heading, description in limitations:
    with st.expander(heading):
        st.write(description)


# =================================================
# FINAL NOTE
# =================================================
st.divider()

st.info(
    """
    **Research prototype disclaimer**

    This dashboard presents experimental machine learning results.
    It is not a medical device, diagnostic system, or substitute for
    professional clinical judgement. Do not use its predictions to
    make patient-care decisions.
    """
)
