from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="Prediction Results | Clinical XAI",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parents[1]
CSS_PATH = BASE_DIR / "assets" / "style.css"

if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as file:
        st.markdown(
            f"<style>{file.read()}</style>",
            unsafe_allow_html=True
        )


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
# RETRIEVE LATEST ASSESSMENT
# -------------------------------------------------
results = st.session_state.get("assessment_results")
observations = st.session_state.get("assessment_observations")
active_suite = st.session_state.get("active_suite", "six_model")

st.title("Prediction Results")
st.write(
    "Comparative analysis of model predictions across the evaluated clinical trajectory."
)

if results is None:
    st.info(
        "No assessment results are available yet. Complete a new patient assessment "
        "or run a demonstration scenario to view model predictions."
    )
    st.page_link("pages/2_New_Assessment.py", label="Go to New Assessment")
    st.stop()


# -------------------------------------------------
# SIX-MODEL SUITE RESULTS DISPLAY
# -------------------------------------------------
if "model_outputs" in results:
    outputs = results["model_outputs"]
    alerts = results["alerts"]

    st.divider()
    st.subheader("Model Comparison: Probabilistic Models")
    st.write(
        "Predicted probabilities represent the estimated likelihood of clinical deterioration "
        "based on the latest hourly observation. Models flag an alert when their score meets or exceeds "
        "their calibrated validation threshold."
    )

    # Row 1: Primary Tabular Models
    col1, col2, col3 = st.columns(3)
    row1_models = [
        ("Original XGBoost", "original_xgboost", col1),
        ("Random Forest", "random_forest", col2),
        ("Logistic Regression", "logistic_regression", col3)
    ]
    for label, key, column in row1_models:
        out = outputs[key]
        prob = out["score"]
        thresh = out["threshold"]
        alert = out["alert"]
        with column:
            st.markdown(f"#### {label}")
            st.metric("Predicted Probability", f"{prob * 100:.2f}%")
            st.progress(min(max(float(prob), 0.0), 1.0))
            st.caption(f"Alert threshold: **{thresh * 100:.2f}%**")
            if alert:
                st.warning("Model Alert: Threshold Crossed")
            else:
                st.success("Below Alert Threshold")

    # Row 2: Clinical-Only & Deep Learning Models
    st.write("")
    col4, col5 = st.columns(2)
    row2_models = [
        ("Clinical-only XGBoost", "clinical_only_xgboost", col4),
        ("Temporal Attention LSTM", "temporal_attention_lstm", col5)
    ]
    for label, key, column in row2_models:
        out = outputs[key]
        prob = out["score"]
        thresh = out["threshold"]
        alert = out["alert"]
        with column:
            st.markdown(f"#### {label}")
            st.metric("Predicted Probability", f"{prob * 100:.2f}%")
            st.progress(min(max(float(prob), 0.0), 1.0))
            st.caption(f"Alert threshold: **{thresh * 100:.2f}%**")
            if alert:
                st.warning("Model Alert: Threshold Crossed")
            else:
                st.success("Below Alert Threshold")

    # Separate section for Linear SVM
    st.divider()
    st.subheader("Model Comparison: Decision-Score Model (Linear SVM)")
    svm_out = outputs["linear_svm"]
    col_svm1, col_svm2 = st.columns([1, 2])
    with col_svm1:
        st.markdown("#### Linear SVM")
        st.metric("Decision Score", f"{svm_out['score']:+.4f}")
        st.caption(f"Decision threshold: **{svm_out['threshold']:+.4f}**")
        if svm_out["alert"]:
            st.warning("Model Alert: Decision Threshold Crossed")
        else:
            st.success("Below Decision Threshold")
    with col_svm2:
        st.markdown("#### Decision Score Interpretation")
        st.markdown(
            """
            <div style="background-color: #ffffff; border: 1px solid #e5e9ec; border-left: 4px solid #367f7a; padding: 20px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                <div style="color: #172b4d; font-family: 'Outfit', sans-serif; font-size: 1.05rem; font-weight: 700; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;">
                    <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: #367f7a;"></span>
                    Linear SVM Decision Score Notice
                </div>
                <div style="color: #263445; font-size: 0.95rem; line-height: 1.6;">
                    LinearSVC outputs a signed, continuous distance from the separating hyperplane via <code style="background-color: #f0f4f8; color: #172b4d; padding: 2px 6px; border-radius: 4px; font-family: monospace;">decision_function()</code>. 
                    Unlike tree-based or logistic models, it does not output a calibrated probability. 
                    Scores greater than the validation threshold indicate a deterioration alert.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Detailed Comparison Table
    st.divider()
    st.subheader("Comprehensive Six-Model Summary Table")
    table_rows = []
    display_order = [
        "original_xgboost",
        "random_forest",
        "logistic_regression",
        "linear_svm",
        "clinical_only_xgboost",
        "temporal_attention_lstm"
    ]
    for key in display_order:
        out = outputs[key]
        if out["output_type"] == "probability":
            score_str = f"{out['score'] * 100:.2f}%"
            thresh_str = f"{out['threshold'] * 100:.2f}%"
            output_type_str = "Probability"
        else:
            score_str = f"{out['score']:+.4f}"
            thresh_str = f"{out['threshold']:+.4f}"
            output_type_str = "Decision Score"

        table_rows.append({
            "Model Name": out["name"],
            "Architecture / Type": "Recurrent Deep Learning" if out["model_type"] == "recurrent_lstm" else "Tabular Classifier",
            "Output Type": output_type_str,
            "Predicted Score": score_str,
            "Decision Threshold": thresh_str,
            "Alert Triggered": "YES (Alert)" if out["alert"] else "No"
        })

    st.dataframe(
        pd.DataFrame(table_rows),
        use_container_width=True,
        hide_index=True
    )

    # Probability vs Threshold Comparison Chart
    st.divider()
    st.subheader("Predicted Probability vs Threshold")
    st.write(
        "Side-by-side comparison of predicted probabilities and alert thresholds for the five probabilistic models."
    )

    prob_keys = [
        ("Original XGBoost", "original_xgboost"),
        ("Random Forest", "random_forest"),
        ("Logistic Regression", "logistic_regression"),
        ("Clinical-only XGBoost", "clinical_only_xgboost"),
        ("Temporal Attention LSTM", "temporal_attention_lstm")
    ]

    model_names = [label for label, _ in prob_keys]
    prob_vals = [outputs[k]["score"] * 100 for _, k in prob_keys]
    thresh_vals = [outputs[k]["threshold"] * 100 for _, k in prob_keys]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4.5))
    rects1 = ax.bar(x - width/2, prob_vals, width, label="Predicted Probability (%)", color="#367f7a")
    rects2 = ax.bar(x + width/2, thresh_vals, width, label="Alert Threshold (%)", color="#8297a8", alpha=0.8)

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Probability vs Threshold Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # Alert Agreement Summary
    st.divider()
    st.subheader("Alert Agreement & Ensemble Consensus")
    total_alerts = sum(alerts.values())

    if total_alerts == 0:
        st.success(
            "**High Negative Consensus (0 / 6 Alerts):** "
            "None of the six models crossed their deterioration threshold."
        )
    elif total_alerts == 6:
        st.warning(
            "**High Alert Consensus (6 / 6 Alerts):** "
            "All six models independently crossed their deterioration threshold."
        )
    elif total_alerts >= 4:
        st.warning(
            f"**Majority Alert Consensus ({total_alerts} / 6 Alerts):** "
            f"{total_alerts} out of 6 models indicate an elevated deterioration risk."
        )
    else:
        st.info(
            f"**Split Decision ({total_alerts} / 6 Alerts):** "
            f"{total_alerts} out of 6 models triggered an alert. Inspect individual model rationales and feature contributions."
        )

# -------------------------------------------------
# LEGACY THREE-MODEL SUITE RESULTS DISPLAY
# -------------------------------------------------
else:
    probabilities = results["probabilities"]
    thresholds = results["thresholds"]
    alerts = results["alerts"]

    st.divider()
    st.subheader("Model Comparison (Three-Model Legacy Suite)")
    col1, col2, col3 = st.columns(3)
    legacy_details = [
        ("Original XGBoost", "original_xgb", col1),
        ("Clinical-only XGBoost", "clinical_xgb", col2),
        ("Temporal Attention LSTM", "lstm", col3)
    ]
    for label, key, column in legacy_details:
        prob = probabilities[key]
        thresh = thresholds[key]
        alert = alerts[key]
        with column:
            st.markdown(f"#### {label}")
            st.metric("Predicted probability", f"{prob * 100:.2f}%")
            st.progress(min(max(float(prob), 0.0), 1.0))
            st.caption(f"Threshold: {thresh * 100:.2f}%")
            if alert:
                st.warning("Model threshold crossed")
            else:
                st.success("Below model threshold")


# -------------------------------------------------
# PATIENT OBSERVATION SUMMARY
# -------------------------------------------------
if observations is not None:
    st.divider()
    st.subheader("Submitted Assessment Trajectory")
    st.write(f"Observations evaluated: **{len(observations)} hours**")

    if "ICULOS" in observations.columns:
        st.write(
            f"ICU Observation Hours: **Hour {int(observations['ICULOS'].min())} – Hour {int(observations['ICULOS'].max())}**"
        )

    with st.expander("View Submitted Hourly Observations Table"):
        st.dataframe(observations, use_container_width=True, hide_index=True)


# -------------------------------------------------
# CLINICAL INTERPRETATION NOTES
# -------------------------------------------------
st.divider()
st.subheader("Important Clinical & Methodological Notes")

st.info(
    "- **Non-Calibrated Output Scales:** Model probabilities and decision scores reflect mathematical loss functions "
    "optimized on retrospective training cohorts. They should not be interpreted as absolute biological probabilities of deterioration.\n"
    "- **Independent Model Dynamics:** Different algorithms weight features differently. For instance, Original XGBoost incorporates "
    "ICU timing context (`ICULOS`, `HospAdmTime`), whereas Clinical-only XGBoost isolates physiological vitals, and the LSTM evaluates multi-hour trajectory dynamics.\n"
    "- **Threshold Crossing vs Clinical Diagnosis:** An alert indicates statistical deviation in vital patterns, not an affirmative medical diagnosis. "
    "Clinical intervention must be guided by patient examination, diagnostic tests, and clinical protocols."
)

col_b1, col_b2 = st.columns(2)
with col_b1:
    if st.button("Start New Assessment", type="primary"):
        st.switch_page("pages/2_New_Assessment.py")
with col_b2:
    if st.button("Explore Feature Explainability (SHAP & Attention)"):
        st.switch_page("pages/4_Explainability.py")
