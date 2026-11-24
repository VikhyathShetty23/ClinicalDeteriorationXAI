from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.explainability import (
    explain_xgboost,
    get_model_feature_importance
)
from utils.preprocessing import (
    validate_observations,
    handle_missing_values,
    create_temporal_features,
    prepare_model_inputs
)


# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="Explainability | Clinical XAI",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parents[1]
CSS_PATH = BASE_DIR / "assets" / "style.css"
SHAP_DIR = BASE_DIR / "experiments" / "six_model_comparison" / "results" / "shap"

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
st.title("Model Explainability & Interpretability")

st.write(
    "Explore how features drive predictions across machine learning models and examine "
    "how the recurrent neural network allocates attention across consecutive hourly observations."
)


# -------------------------------------------------
# RETRIEVE ASSESSMENT DATA
# -------------------------------------------------
research_models = st.session_state.get("research_models")
legacy_models = st.session_state.get("models")
observations = st.session_state.get("assessment_observations")
results = st.session_state.get("assessment_results")
active_suite = st.session_state.get("active_suite", "six_model")


# =====================================================
# SECTION 1: INDIVIDUAL PATIENT SHAP EXPLANATION
# =====================================================
st.divider()
st.subheader("1. Individual Patient Assessment Explanations (Local SHAP)")

if observations is None or results is None:
    st.info(
        "No active patient assessment found. Run a new assessment or demo scenario in **New Assessment** "
        "to compute case-level SHAP attributions and patient-specific LSTM temporal attention."
    )
    st.page_link("pages/2_New_Assessment.py", label="Go to New Assessment to Run a Patient")
else:
    st.write(
        "SHapley Additive exPlanations (SHAP) decompose the model output into additive feature contributions. "
        "Positive values push the prediction toward clinical deterioration; negative values push it lower."
    )

    # Determine which models to explain
    if active_suite == "six_model" and research_models is not None:
        model_choice = st.selectbox(
            "Select Model for Individual SHAP Explanation",
            ["Original XGBoost (48 features)", "Clinical-only XGBoost (46 features)"]
        )

        if "Original" in model_choice:
            art = research_models["tabular"]["original_xgboost"]
        else:
            art = research_models["tabular"]["clinical_only_xgboost"]

        selected_model = art["model"]
        feature_names = art["feature_columns"]
        imputer = art["imputer"]
        scaler = art["scaler"]

        # Prepare input for selected model
        try:
            val_df = validate_observations(observations)
            val_df = handle_missing_values(val_df)
            val_df = create_temporal_features(val_df).replace([np.inf, -np.inf], np.nan)
            X = val_df[feature_names].copy()
            X = imputer.transform(X)
            if scaler is not None:
                X = scaler.transform(X)
            selected_input = X[-1:].astype(np.float32)
        except Exception as e:
            st.error(f"Error preparing features for SHAP: {e}")
            selected_input = None

    else:
        # Legacy fallback
        model_choice = st.selectbox(
            "Select Model for Individual SHAP Explanation",
            ["Original XGBoost", "Clinical-only XGBoost"]
        )
        try:
            m_inputs = prepare_model_inputs(
                observations,
                legacy_models["config"],
                legacy_models["xgb_imputer"],
                legacy_models["clinical_imputer"],
                legacy_models["lstm_imputer"],
                legacy_models["lstm_scaler"]
            )
            if "Original" in model_choice:
                selected_model = legacy_models["original_xgb"]
                selected_input = m_inputs["original_xgb"]
                feature_names = legacy_models["config"]["xgb_features"]
            else:
                selected_model = legacy_models["clinical_xgb"]
                selected_input = m_inputs["clinical_xgb"]
                feature_names = legacy_models["config"]["clinical_features"]
        except Exception as e:
            st.error(f"Error preparing legacy inputs: {e}")
            selected_input = None

    if selected_input is not None:
        try:
            explanation = explain_xgboost(selected_model, selected_input, feature_names)
            contributions = explanation["contributions"]
            base_val = explanation["base_value"]

            st.write(
                f"**Base Value (Expected log-odds):** `{base_val:.4f}` | "
                f"**Assessment Feature Count:** `{len(feature_names)}`"
            )

            # Horizontal Bar Chart for Top 15 SHAP features
            top_15 = contributions[:15]
            c_df = pd.DataFrame(top_15)

            fig, ax = plt.subplots(figsize=(10, 6))
            vals = c_df["value"].to_numpy()
            feats = c_df["feature"].to_numpy()

            # Teal (#367f7a) for positive (higher deterioration risk), terracotta (#c47b55) for negative
            colors = ["#367f7a" if v >= 0 else "#c47b55" for v in vals]

            ax.barh(feats[::-1], vals[::-1], color=colors[::-1])
            ax.axvline(0, color="#263445", linewidth=0.8, linestyle="--")
            ax.set_xlabel("SHAP Value (Impact on Model Output Scale)")
            ax.set_ylabel("Feature")
            ax.set_title(f"Top 15 Individual SHAP Contributions — {model_choice}")
            ax.grid(axis="x", alpha=0.3)
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Teal bars indicate features increasing the predicted deterioration score; "
                "terracotta bars indicate features driving the score lower. "
                "Values are additive on the log-odds/margin scale, not absolute probability percentages."
            )

            with st.expander("View Full List of All Individual SHAP Contributions"):
                st.dataframe(pd.DataFrame(contributions), use_container_width=True, hide_index=True)

        except Exception as err:
            st.error(f"Unable to compute local SHAP explanation: {err}")


# =====================================================
# SECTION 2: BUILT-IN MODEL FEATURE IMPORTANCE
# =====================================================
st.divider()
st.subheader("2. Model-Level Feature Importance (XGBoost Split/Gain)")

st.write(
    "Built-in feature importance reflects how frequently and effectively features were used to split "
    "decision trees across the entire training ensemble. This is a global model-level property, distinct from case-level SHAP."
)

if research_models is not None and "tabular" in research_models:
    try:
        xgb_art = research_models["tabular"]["original_xgboost"]
        importance_list = get_model_feature_importance(xgb_art["model"], xgb_art["feature_columns"])
        top_imp = importance_list[:15]
        imp_df = pd.DataFrame(top_imp)

        fig_imp, ax_imp = plt.subplots(figsize=(10, 5))
        ax_imp.barh(imp_df["feature"].to_numpy()[::-1], imp_df["importance"].to_numpy()[::-1], color="#367f7a")
        ax_imp.set_xlabel("Feature Importance (Gain / Split Weight)")
        ax_imp.set_ylabel("Feature")
        ax_imp.set_title("Top 15 Global Feature Importance (Original XGBoost)")
        ax_imp.grid(axis="x", alpha=0.3)
        fig_imp.tight_layout()

        st.pyplot(fig_imp)
        plt.close(fig_imp)
    except Exception as e:
        st.warning(f"Could not render built-in feature importance: {e}")


# =====================================================
# SECTION 3: LSTM TEMPORAL ATTENTION WEIGHTS
# =====================================================
st.divider()
st.subheader("3. LSTM Temporal Attention Distribution")

st.write(
    "The Temporal Attention LSTM processes a sequence of six consecutive hourly observations. "
    "A self-attention layer assigns dynamic weights summing to 1.0 across the six time steps, "
    "highlighting which hourly observations the recurrent network emphasized."
)

if results is not None and "lstm_attention" in results:
    attn_weights = results["lstm_attention"]
    if len(attn_weights) == 6 and observations is not None:
        hours = observations["ICULOS"].astype(int).tolist()[-6:]
        attn_df = pd.DataFrame({
            "ICU Hour": [f"Hour {h}" for h in hours],
            "Attention Weight": attn_weights
        })

        fig_attn, ax_attn = plt.subplots(figsize=(9, 4))
        ax_attn.bar(attn_df["ICU Hour"], attn_df["Attention Weight"], color="#367f7a", width=0.5)
        ax_attn.set_ylabel("Attention Weight (Sum = 1.0)")
        ax_attn.set_xlabel("Consecutive Observation Hour")
        ax_attn.set_title("Temporal Attention Distribution Across Evaluated Hours")
        ax_attn.set_ylim(0, max(attn_weights) * 1.25)
        ax_attn.grid(axis="y", alpha=0.3)
        fig_attn.tight_layout()

        st.pyplot(fig_attn)
        plt.close(fig_attn)

        st.dataframe(attn_df, use_container_width=True, hide_index=True)
        st.caption(
            "Higher attention indicates greater relative weighting by the LSTM classification head. "
            "Attention reflects computational focus within the sequence model, not physiological causality."
        )
    else:
        st.info("LSTM attention weights are available after running a 6-hour patient sequence.")
else:
    st.info("Run an assessment above to visualize case-specific LSTM temporal attention weights.")


# =====================================================
# SECTION 4: GLOBAL SHAP RESEARCH FINDINGS
# =====================================================
st.divider()
st.subheader("4. Global SHAP Research Findings (Cohort-Wide Analysis)")

st.write(
    "The research investigation performed comprehensive SHAP attribution across test patients "
    "using `shap.TreeExplainer` on the Original XGBoost model."
)

# A. Global Mean Absolute SHAP CSV
shap_csv_path = SHAP_DIR / "shap_feature_importance.csv"
if shap_csv_path.exists():
    try:
        global_shap_df = pd.read_csv(shap_csv_path)
        col_g1, col_g2 = st.columns([1, 1])
        with col_g1:
            st.markdown("#### Top Features by Mean Absolute SHAP")
            st.dataframe(
                global_shap_df.head(15).style.format({"Mean_Absolute_SHAP": "{:.4f}"}),
                use_container_width=True,
                hide_index=True
            )
        with col_g2:
            st.info(
                "**Cohort-Level Finding:**\n\n"
                "- **`ICULOS` (ICU Length of Stay):** Strongest overall predictor (Mean |SHAP| = 0.6239), "
                "indicating that hospital dwell time carries substantial predictive weight.\n"
                "- **`HospAdmTime`:** Second strongest predictor (0.3583), capturing emergency vs elective admission timing.\n"
                "- **`Age`:** Third strongest (0.2646), reflecting baseline physiological vulnerability.\n"
                "- **Vital Trends (`Temp_mean_6h`, `Temp`, `HR_mean_6h`):** Leading physiological drivers."
            )
    except Exception as e:
        st.warning(f"Could not load global SHAP CSV: {e}")

# B. Verified Research Plots
st.markdown("#### Cohort-Level SHAP Visualizations from Research Experiments")

tabs = st.tabs(["Summary Beeswarm Plot", "Feature Importance Ranking", "Representative Waterfall Example"])

with tabs[0]:
    beeswarm_path = SHAP_DIR / "shap_summary_beeswarm.png"
    if beeswarm_path.exists():
        st.image(
            str(beeswarm_path),
            caption="SHAP Summary Beeswarm Plot: Displays the distribution of SHAP values and feature value coloring across test patients.",
            use_container_width=True
        )
    else:
        st.info("Beeswarm plot image not found at expected path.")

with tabs[1]:
    feat_imp_path = SHAP_DIR / "shap_feature_importance.png"
    if feat_imp_path.exists():
        st.image(
            str(feat_imp_path),
            caption="Global SHAP Feature Importance: Ranked by mean absolute SHAP value across the evaluation cohort.",
            use_container_width=True
        )
    else:
        st.info("Feature importance plot image not found at expected path.")

with tabs[2]:
    waterfall_path = SHAP_DIR / "shap_waterfall_example.png"
    if waterfall_path.exists():
        st.image(
            str(waterfall_path),
            caption="SHAP Waterfall Example: Illustrates step-by-step feature contributions pushing a patient's prediction from base expected value to final alert score.",
            use_container_width=True
        )
    else:
        st.info("Waterfall plot image not found at expected path.")


# =====================================================
# SECTION 5: METHODOLOGICAL LIMITATIONS
# =====================================================
st.divider()
st.subheader("Methodological & Clinical Limitations")

st.info(
    "1. **Non-Causal Interpretability:** SHAP values and LSTM attention weights describe algorithmic association "
    "and information routing within trained mathematical estimators. They do not demonstrate direct biological causality.\n"
    "2. **Confounding by Operational Timing:** As shown in feature ablation studies, operational variables (`ICULOS`, `HospAdmTime`) "
    "materially contribute to Original XGBoost predictions, reflecting clinical practice patterns rather than pure physiology.\n"
    "3. **Decision-Support Scope:** All explainability visualizers are provided for research transparency and must not be used "
    "as the sole basis for clinical treatment decisions."
)
