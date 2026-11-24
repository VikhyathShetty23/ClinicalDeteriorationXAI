from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="Trends & Research | Clinical XAI",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parents[1]
CSS_PATH = BASE_DIR / "assets" / "style.css"
RESEARCH_DIR = BASE_DIR / "experiments" / "six_model_comparison"
RESULTS_DIR = RESEARCH_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
LEAD_TIME_DIR = RESEARCH_DIR / "lead_time"
ABLATION_DIR = RESEARCH_DIR / "ablation" / "results"

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
# PAGE HEADING
# -------------------------------------------------
st.title("Trends & Comprehensive Research Dashboard")

st.write(
    "Explore individual patient vital-sign trajectories and review empirical findings "
    "from the six-model research experiments, including common test benchmarks, paired bootstrap "
    "hypothesis testing, feature ablation, lead-time early warning, alert episode dynamics, and threshold sensitivity."
)


# =====================================================
# SECTION 1: PATIENT VITAL-SIGN TRAJECTORIES
# =====================================================
st.divider()
st.subheader("1. Patient Vital-Sign Trajectory Trends")

observations = st.session_state.get("assessment_observations")

if observations is not None:
    vital_features = ["HR", "O2Sat", "Temp", "SBP", "MAP", "Resp"]
    available_features = [col for col in vital_features if col in observations.columns]

    col_v1, col_v2 = st.columns([1, 2])
    with col_v1:
        selected_vital = st.selectbox("Select Vital Sign to Plot", available_features)

    fig_vital, ax_vital = plt.subplots(figsize=(10, 3.8))
    ax_vital.plot(
        observations["ICULOS"],
        observations[selected_vital],
        marker="o",
        linewidth=2,
        color="#367f7a"
    )
    ax_vital.set_xlabel("ICU Observation Hour (ICULOS)")
    ax_vital.set_ylabel(selected_vital)
    ax_vital.set_title(f"{selected_vital} Longitudinal Trajectory")
    ax_vital.grid(alpha=0.3)
    fig_vital.tight_layout()

    st.pyplot(fig_vital)
    plt.close(fig_vital)

    with st.expander("View Submitted Vital-Sign Records Table"):
        st.dataframe(observations[["ICULOS"] + available_features], use_container_width=True, hide_index=True)
else:
    st.info(
        "No active patient assessment found. Run a new assessment or demo scenario in **New Assessment** "
        "to plot case-specific vital sign trajectories."
    )


# =====================================================
# SECTION 2: SIX-MODEL COMPARATIVE BENCHMARKS
# =====================================================
st.divider()
st.subheader("2. Six-Model Comparative Performance (Common Test Benchmark)")

st.write(
    "All six models were evaluated on the common held-out test cohort containing "
    "**58,132 eligible hourly endpoints** across **1,721 patients** with **1,300 positive deterioration labels** "
    "(positive prevalence ~2.237%)."
)

common_test_path = RESULTS_DIR / "common_test_model_comparison.csv"
if common_test_path.exists():
    try:
        common_df = pd.read_csv(common_test_path)
        st.dataframe(
            common_df.style.format({
                "Threshold": "{:.4f}",
                "AUROC": "{:.4f}",
                "AUPRC": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}",
                "TN": "{:,}",
                "FP": "{:,}",
                "FN": "{:,}",
                "TP": "{:,}"
            }),
            use_container_width=True,
            hide_index=True
        )

        # Bar charts for AUROC and AUPRC
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_auroc, ax_auroc = plt.subplots(figsize=(7, 4))
            ax_auroc.barh(common_df["Model"][::-1], common_df["AUROC"][::-1], color="#367f7a")
            ax_auroc.set_xlim(0, 1.0)
            ax_auroc.set_xlabel("AUROC")
            ax_auroc.set_title("Area Under ROC Curve (Common Test Set)")
            ax_auroc.grid(axis="x", alpha=0.3)
            fig_auroc.tight_layout()
            st.pyplot(fig_auroc)
            plt.close(fig_auroc)

        with col_c2:
            fig_auprc, ax_auprc = plt.subplots(figsize=(7, 4))
            ax_auprc.barh(common_df["Model"][::-1], common_df["AUPRC"][::-1], color="#8297a8")
            ax_auprc.set_xlabel("AUPRC")
            ax_auprc.set_title("Area Under PR Curve (Common Test Set)")
            ax_auprc.grid(axis="x", alpha=0.3)
            fig_auprc.tight_layout()
            st.pyplot(fig_auprc)
            plt.close(fig_auprc)

    except Exception as e:
        st.warning(f"Could not load common test comparison: {e}")
else:
    st.warning("common_test_model_comparison.csv not found.")

# Visualizer for Research Benchmark Plots
st.markdown("#### Research Experiment Visualizations")
plot_tabs = st.tabs(["ROC Curves", "Precision-Recall Curves", "Metric Comparison", "Confusion Matrices"])

with plot_tabs[0]:
    roc_p = PLOTS_DIR / "roc_curve.png"
    if roc_p.exists():
        st.image(str(roc_p), caption="ROC Curves comparing all six models on the common test cohort.", use_container_width=True)
    else:
        st.info("ROC curve plot not found.")

with plot_tabs[1]:
    pr_p = PLOTS_DIR / "precision_recall_curve.png"
    if pr_p.exists():
        st.image(str(pr_p), caption="Precision-Recall Curves illustrating positive class discrimination.", use_container_width=True)
    else:
        st.info("Precision-Recall curve plot not found.")

with plot_tabs[2]:
    mc_p = PLOTS_DIR / "metric_comparison.png"
    if mc_p.exists():
        st.image(str(mc_p), caption="Comprehensive metric comparison across AUROC, AUPRC, Precision, Recall, and F1.", use_container_width=True)
    else:
        st.info("Metric comparison plot not found.")

with plot_tabs[3]:
    cm_p = PLOTS_DIR / "confusion_matrices.png"
    if cm_p.exists():
        st.image(str(cm_p), caption="Normalized Confusion Matrices across all evaluated models at their selected operating thresholds.", use_container_width=True)
    else:
        st.info("Confusion matrices plot not found.")


# =====================================================
# SECTION 3: PAIRED BOOTSTRAP STATISTICAL COMPARISON
# =====================================================
st.divider()
st.subheader("3. Statistical Significance (Paired Bootstrap Hypothesis Testing)")

st.write(
    "To determine whether differences between Original XGBoost and other models were statistically significant, "
    "paired non-parametric bootstrap resampling (1,000 bootstrap iterations) was conducted on the common test cohort."
)

bootstrap_path = RESULTS_DIR / "paired_bootstrap_comparison.csv"
if bootstrap_path.exists():
    try:
        boot_df = pd.read_csv(bootstrap_path)
        st.dataframe(
            boot_df.style.format({
                "AUROC_Difference": "{:+.4f}",
                "AUROC_CI_Lower": "{:+.4f}",
                "AUROC_CI_Upper": "{:+.4f}",
                "AUPRC_Difference": "{:+.4f}",
                "AUPRC_CI_Lower": "{:+.4f}",
                "AUPRC_CI_Upper": "{:+.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "**Key Statistical Insights:**\n\n"
            "- **Original XGBoost vs Clinical-only XGBoost:** AUROC difference is **+0.0857** (95% CI: `[+0.0503, +0.1195]`). "
            "Because the confidence interval **excludes zero**, the performance advantage of including timing variables is statistically significant.\n"
            "- **Original XGBoost vs Temporal Attention LSTM:** AUROC difference is **+0.1051** (95% CI: `[+0.0628, +0.1476]`). "
            "Confidence interval strictly excludes zero.\n"
            "- **Original XGBoost vs Random Forest, Logistic Regression, Linear SVM:** In all three cases, the 95% confidence intervals "
            "**cross zero**, indicating that differences in discrimination among the 48-feature tabular models are not statistically significant at $\\alpha = 0.05$."
        )
    except Exception as e:
        st.warning(f"Could not load paired bootstrap comparison: {e}")
else:
    st.warning("paired_bootstrap_comparison.csv not found.")


# =====================================================
# SECTION 4: FEATURE ABLATION EXPERIMENTS
# =====================================================
st.divider()
st.subheader("4. Feature Ablation Analysis (Quantifying Timing Confounders)")

st.write(
    "Systematic feature ablation experiments isolated the exact impact of ICU timing variables "
    "(`ICULOS` — ICU length of stay, and `HospAdmTime` — pre-ICU hospital admission offset) on XGBoost performance."
)

ablation_path = ABLATION_DIR / "ablation_comparison.csv"
if ablation_path.exists():
    try:
        abl_df = pd.read_csv(ablation_path)
        col_ab1, col_ab2 = st.columns([1, 1])

        with col_ab1:
            st.dataframe(
                abl_df[["Model", "AUROC", "AUPRC", "Precision", "Recall", "F1", "Threshold"]].style.format({
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

        with col_ab2:
            st.info(
                "**Ablation Findings:**\n\n"
                "- **Removing `HospAdmTime` alone:** Minimal impact. AUROC remains virtually identical (**0.7910** vs 0.7912 baseline).\n"
                "- **Removing `ICULOS` alone:** Dramatic performance drop to **0.7167** AUROC and **0.0613** AUPRC.\n"
                "- **Removing Both (`ICULOS` & `HospAdmTime`):** AUROC drops to **0.7055** and AUPRC to **0.0621**.\n\n"
                "**Conclusion:** ICU length of stay (`ICULOS`) is the primary driver of tabular model inflation. "
                "Patients who remain in the ICU longer accumulate higher baseline risk, creating an operational shortcut that models exploit."
            )
    except Exception as e:
        st.warning(f"Could not load ablation results: {e}")
else:
    st.warning("ablation_comparison.csv not found.")


# =====================================================
# SECTION 5: EARLY-WARNING LEAD-TIME ANALYSIS
# =====================================================
st.divider()
st.subheader("5. Early-Warning Lead-Time Evaluation")

st.write(
    "To assess clinical utility before overt onset, alerts generated by the Original XGBoost model "
    "were evaluated at fixed horizons (6, 4, and 2 hours prior to estimated clinical deterioration) across the full test cohort."
)

lead_time_path = LEAD_TIME_DIR / "lead_time_summary_full_test.csv"
if lead_time_path.exists():
    try:
        lt_df = pd.read_csv(lead_time_path)
        st.dataframe(
            lt_df.style.format({
                "Total_Sepsis_Patients": "{:d}",
                "Eligible_Patients": "{:d}",
                "Unavailable_Patients": "{:d}",
                "Detected_Patients": "{:d}",
                "Missed_Eligible_Patients": "{:d}",
                "Detection_Rate_Among_Eligible_Pct": "{:.2f}%",
                "Detection_Rate_Among_All_Sepsis_Patients_Pct": "{:.2f}%",
                "Threshold": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        fig_lt, ax_lt = plt.subplots(figsize=(8, 3.5))
        horizons = [f"{h}h Pre-Onset" for h in lt_df["Lead_Time_Hours"]]
        rates = lt_df["Detection_Rate_Among_Eligible_Pct"]
        ax_lt.bar(horizons, rates, color="#367f7a", width=0.45)
        ax_lt.set_ylabel("Detection Rate Among Eligible Patients (%)")
        ax_lt.set_title("Early Deterioration Detection Rate Across Lead-Time Horizons")
        ax_lt.set_ylim(0, 50)
        ax_lt.grid(axis="y", alpha=0.3)
        fig_lt.tight_layout()
        st.pyplot(fig_lt)
        plt.close(fig_lt)

    except Exception as e:
        st.warning(f"Could not load lead time summary: {e}")
else:
    st.warning("lead_time_summary_full_test.csv not found.")


# =====================================================
# SECTION 6: CONTINUOUS EARLY-WARNING & FALSE ALARM RATE
# =====================================================
st.divider()
st.subheader("6. Continuous Early-Warning Window & False Alarm Evaluation")

st.write(
    "Analysis of alert dynamics in a continuous clinical window from 2 to 6 hours prior to estimated onset, "
    "benchmarked against non-deteriorating control patients."
)

cont_path = LEAD_TIME_DIR / "continuous_warning_summary.csv"
if cont_path.exists():
    try:
        cont_df = pd.read_csv(cont_path)
        row = cont_df.iloc[0]

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Eligible Pre-Onset Detection Rate", f"{row['Detection_Rate_Eligible_Pct']:.2f}%", "45 of 128 patients")
        with col_m2:
            st.metric("Mean Earliest Warning", f"{row['Mean_Earliest_Warning_Hours']:.2f} hours", "Median: 6.0 hours pre-onset")
        with col_m3:
            st.metric("Non-Sepsis Patient Alert Rate", f"{row['Non_Sepsis_Patient_Alert_Rate_Pct']:.2f}%", "108 of 1,573 controls")
        with col_m4:
            st.metric("False Alert Frequency", f"{row['False_Alert_Hours_Per_1000_Observed_Hours']:.2f}", "Hours per 1,000 patient-hours")

        st.caption(
            f"**Early Warning Breakdown:** Among detected patients, {int(row['Warnings_At_6_Hours'])} triggered alerts at exactly 6 hours pre-onset, "
            f"{int(row['Warnings_4_to_6_Hours'])} between 4–6 hours, and {int(row['Warnings_2_to_4_Hours'])} between 2–4 hours pre-onset."
        )
    except Exception as e:
        st.warning(f"Could not load continuous warning summary: {e}")
else:
    st.warning("continuous_warning_summary.csv not found.")


# =====================================================
# SECTION 7: ALERT EPISODE DYNAMICS
# =====================================================
st.divider()
st.subheader("7. Alert Episode Dynamics (Duration & Sustained Burden)")

st.write(
    "To differentiate transient false alarms from persistent clinical deterioration, alert sequences were grouped "
    "into continuous episodes (consecutive hours with active alerts)."
)

ep_path = LEAD_TIME_DIR / "alert_episode_summary.csv"
if ep_path.exists():
    try:
        ep_df = pd.read_csv(ep_path)
        st.dataframe(
            ep_df.style.format({
                "Patients": "{:,}",
                "Observed_Hours": "{:,}",
                "Patients_With_Alerts": "{:,}",
                "Total_Episodes": "{:,}",
                "Isolated_Episodes": "{:,}",
                "Sustained_Episodes": "{:,}",
                "Mean_Duration_Hours": "{:.2f}",
                "Median_Duration_Hours": "{:.1f}",
                "Episodes_Per_24_Patient_Hours": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "**Key Alert Episode Observations:**\n\n"
            "- **Sepsis Patients (Pre-Onset Window):** Average episode duration is **3.54 hours** (median 5.0h), with **42 out of 52 episodes (80.8%)** being sustained (≥ 2 consecutive hours).\n"
            "- **Non-Sepsis Controls:** Average episode duration is **1.98 hours** (median 1.0h), with **55.5%** being isolated single-hour spikes.\n"
            "- **Clinical Escalation Principle:** Requiring sustained alerts over consecutive hours significantly filters out transient measurement artifacts without sacrificing pre-onset detection."
        )
    except Exception as e:
        st.warning(f"Could not load alert episode summary: {e}")
else:
    st.warning("alert_episode_summary.csv not found.")


# =====================================================
# SECTION 8: OPERATING THRESHOLD SENSITIVITY
# =====================================================
st.divider()
st.subheader("8. Operating Threshold Sensitivity (Trade-Off Curve)")

st.write(
    "Evaluating the clinical trade-off between pre-onset detection sensitivity and false alert burden "
    "across different decision thresholds."
)

sens_path = LEAD_TIME_DIR / "threshold_sensitivity_results.csv"
if sens_path.exists():
    try:
        sens_df = pd.read_csv(sens_path)
        st.dataframe(
            sens_df.style.format({
                "Threshold": "{:.4f}",
                "Total_Sepsis_Patients": "{:d}",
                "Eligible_Sepsis_Patients": "{:d}",
                "Detected_Sepsis_Patients": "{:d}",
                "Detection_Rate_Eligible_Pct": "{:.2f}%",
                "Non_Sepsis_Patients": "{:d}",
                "Non_Sepsis_Patients_With_Alerts": "{:d}",
                "Non_Sepsis_Patient_Alert_Rate_Pct": "{:.2f}%",
                "False_Alert_Hours_Per_1000_Hours": "{:.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        # Sensitivity curve
        fig_sens, ax1 = plt.subplots(figsize=(9, 4.2))
        ax2 = ax1.twinx()

        color1 = "#367f7a"
        color2 = "#c47b55"

        ax1.plot(sens_df["Threshold"], sens_df["Detection_Rate_Eligible_Pct"], marker="o", color=color1, linewidth=2, label="Pre-Onset Detection Rate (%)")
        ax2.plot(sens_df["Threshold"], sens_df["False_Alert_Hours_Per_1000_Hours"], marker="s", color=color2, linewidth=2, linestyle="--", label="False Alert Rate / 1k Hours")

        ax1.set_xlabel("Decision Threshold")
        ax1.set_ylabel("Detection Rate Among Eligible Patients (%)", color=color1)
        ax2.set_ylabel("False Alert Hours per 1,000 Observed Hours", color=color2)
        ax1.set_title("Threshold Operating Sensitivity vs Alarm Fatigue Trade-Off")
        ax1.grid(alpha=0.3)
        fig_sens.tight_layout()

        st.pyplot(fig_sens)
        plt.close(fig_sens)

    except Exception as e:
        st.warning(f"Could not load threshold sensitivity results: {e}")
else:
    st.warning("threshold_sensitivity_results.csv not found.")


# =====================================================
# SECTION 9: RESEARCH LIMITATIONS
# =====================================================
st.divider()
st.subheader("9. Research Limitations")

limitations = [
    (
        "Severe Class Imbalance",
        "Positive deterioration labels account for only ~2.24% of hourly observations in the test cohort. "
        "Consequently, precision and AUPRC metrics are substantially lower than AUROC, reflecting the high penalty for false positives in imbalanced populations."
    ),
    (
        "Confounding Operational Timing Variables",
        "As demonstrated by feature ablation and bootstrap analysis, the Original XGBoost model gains considerable predictive leverage "
        "from ICU length of stay (ICULOS). While operationally predictive, it creates risk of shortcut learning when applied across differing clinical settings."
    ),
    (
        "Retrospective Dataset Scope",
        "The models were trained on an experimental subset of the PhysioNet 2019 dataset and may not reflect patient demographics, "
        "admission protocols, or measurement frequencies of other hospitals."
    ),
    (
        "Observational Measurement Artifacts",
        "Vital-sign documentation frequencies vary widely between stable and deteriorating patients. "
        "While explicit missingness indicators capture this clinical reality, changes in institutional monitoring policies could shift model behavior."
    ),
    (
        "Lack of Prospective Clinical Validation",
        "These models are retrospective research prototypes. They have not been validated in live electronic health record (EHR) environments, "
        "prospective randomized clinical trials, or clinical utility studies."
    )
]

for title, desc in limitations:
    with st.expander(title):
        st.write(desc)


# -------------------------------------------------
# DISCLAIMER
# -------------------------------------------------
st.divider()
st.info(
    "**Research Dashboard Notice:** All tables, plots, and metrics displayed above are reproducible outputs "
    "from the six-model research experiments conducted on the PhysioNet 2019 sepsis prediction cohort. "
    "They are presented for academic transparency and peer review."
)
