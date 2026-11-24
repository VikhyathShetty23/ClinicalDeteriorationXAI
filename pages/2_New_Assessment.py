from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

from utils.research_model_loader import load_research_models
from utils.research_prediction import predict_six_models
from utils.model_loader import load_models
from utils.preprocessing import prepare_model_inputs
from utils.prediction import predict_all


# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="New Assessment | Clinical XAI",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parents[1]
CSS_PATH = BASE_DIR / "assets" / "style.css"
TEMPLATE_PATH = BASE_DIR / "sample_patient_template.csv"

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
# RETRIEVE LOADED MODELS
# -------------------------------------------------
research_models = st.session_state.get("research_models")
legacy_models = st.session_state.get("models")

if research_models is None:
    try:
        research_models = load_research_models()
        st.session_state["research_models"] = research_models
    except Exception as e:
        st.error(f"Unable to load six-model research suite: {e}")
        st.stop()

if legacy_models is None:
    try:
        legacy_models = load_models()
        st.session_state["models"] = legacy_models
    except Exception:
        pass


# -------------------------------------------------
# PAGE HEADING
# -------------------------------------------------
st.title("New Patient Assessment")

st.write(
    "Evaluate consecutive hourly patient vital-sign observations across trained "
    "temporal machine learning and deep learning models. Enter observations manually, "
    "upload a structured CSV, or evaluate synthetic clinical demo scenarios."
)

st.info(
    "**Research prototype only:** Model outputs are experimental decision-support estimates "
    "and do not constitute clinical diagnoses or treatment recommendations."
)


# -------------------------------------------------
# MODEL SUITE SELECTION
# -------------------------------------------------
col_suite, _ = st.columns([2, 1])
with col_suite:
    model_suite = st.radio(
        "Select Model Suite",
        [
            "Six-Model Research Suite (Complete)",
            "Three-Model Legacy Suite"
        ],
        help="Six-Model Suite includes Original XGBoost, Random Forest, Logistic Regression, Linear SVM, Clinical-only XGBoost, and Temporal Attention LSTM."
    )

is_six_model = "Six-Model" in model_suite


# -------------------------------------------------
# INPUT METHOD SELECTION
# -------------------------------------------------
input_method = st.radio(
    "Choose Input Method",
    ["Manual Entry", "CSV Upload", "Demo Mode"],
    horizontal=True
)


# -------------------------------------------------
# ASSESSMENT RUNNER FUNCTION
# -------------------------------------------------
def execute_assessment(observations_df: pd.DataFrame):
    """
    Execute predictions using the selected model suite.
    """
    if is_six_model:
        # Six-model research pipeline
        results = predict_six_models(research_models, observations_df)
        st.session_state["assessment_results"] = results
        st.session_state["assessment_observations"] = observations_df.copy()
        st.session_state["assessment_processed"] = results["processed_data"].copy()
        st.session_state["active_suite"] = "six_model"

        st.success("Six-model assessment completed successfully.")
        st.write("### Prediction Summary (Six Research Models)")

        # Display probability-producing models in top row
        st.markdown("#### Probability-Based Models")
        col1, col2, col3 = st.columns(3)
        prob_cards = [
            ("Original XGBoost", "original_xgboost", col1),
            ("Random Forest", "random_forest", col2),
            ("Logistic Regression", "logistic_regression", col3)
        ]
        for label, key, col in prob_cards:
            out = results["model_outputs"][key]
            with col:
                st.metric(label, f"{out['score'] * 100:.2f}%")
                if out["alert"]:
                    st.warning(f"Alert: Threshold ({out['threshold'] * 100:.2f}%) Crossed")
                else:
                    st.success(f"Below Threshold ({out['threshold'] * 100:.2f}%)")

        col4, col5 = st.columns(2)
        prob_cards_row2 = [
            ("Clinical-only XGBoost", "clinical_only_xgboost", col4),
            ("Temporal Attention LSTM", "temporal_attention_lstm", col5)
        ]
        for label, key, col in prob_cards_row2:
            out = results["model_outputs"][key]
            with col:
                st.metric(label, f"{out['score'] * 100:.2f}%")
                if out["alert"]:
                    st.warning(f"Alert: Threshold ({out['threshold'] * 100:.2f}%) Crossed")
                else:
                    st.success(f"Below Threshold ({out['threshold'] * 100:.2f}%)")

        # Display Linear SVM separately
        st.markdown("#### Decision-Score Model (Linear SVM)")
        svm_out = results["model_outputs"]["linear_svm"]
        col_svm, col_svm_note = st.columns([1, 2])
        with col_svm:
            st.metric("Linear SVM Decision Score", f"{svm_out['score']:+.4f}")
            if svm_out["alert"]:
                st.warning(f"Alert: Threshold ({svm_out['threshold']:.4f}) Crossed")
            else:
                st.success(f"Below Threshold ({svm_out['threshold']:.4f})")
        with col_svm_note:
            st.caption(
                "**Linear SVM Note:** LinearSVC produces an unbounded continuous decision score via `decision_function()`. "
                "It is not a calibrated probability and is evaluated against its validation decision threshold."
            )

    else:
        # Legacy three-model pipeline
        if legacy_models is None:
            st.error("Legacy models are not loaded.")
            return

        model_inputs = prepare_model_inputs(
            observations_df,
            legacy_models["config"],
            legacy_models["xgb_imputer"],
            legacy_models["clinical_imputer"],
            legacy_models["lstm_imputer"],
            legacy_models["lstm_scaler"]
        )

        results = predict_all(legacy_models, model_inputs)
        st.session_state["assessment_results"] = results
        st.session_state["assessment_observations"] = observations_df.copy()
        st.session_state["assessment_processed"] = model_inputs["processed_data"].copy()
        st.session_state["active_suite"] = "legacy"

        st.success("Three-model assessment completed successfully.")
        st.write("### Prediction Summary (Three-Model Legacy Suite)")

        col1, col2, col3 = st.columns(3)
        legacy_cards = [
            ("Original XGBoost", "original_xgb", col1),
            ("Clinical-only XGBoost", "clinical_xgb", col2),
            ("Temporal Attention LSTM", "lstm", col3)
        ]
        for label, key, column in legacy_cards:
            with column:
                probability = results["probabilities"][key]
                alert = results["alerts"][key]
                st.metric(label, f"{probability * 100:.2f}%")
                if alert:
                    st.warning("Model threshold crossed")
                else:
                    st.success("Below model threshold")

    st.caption(
        "Model scores reflect experimental pattern recognition on vital-sign dynamics. "
        "They do not replace comprehensive clinical evaluation."
    )


# =====================================================
# 1. MANUAL ENTRY
# =====================================================
if input_method == "Manual Entry":
    st.subheader("Patient Demographics & Context")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=120, value=65)

    with col2:
        gender_label = st.selectbox("Gender", ["Female", "Male"])

    with col3:
        hospital_admission_time = st.number_input(
            "HospAdmTime",
            value=-5.0,
            step=1.0,
            help="Hours elapsed between hospital admission and ICU admission."
        )

    st.subheader("Consecutive Hourly Observations (Minimum 6 Hours Required)")
    start_iculos = st.number_input(
        "Starting ICU observation hour (ICULOS)",
        min_value=1,
        value=10,
        step=1
    )

    st.caption(
        "Enter six consecutive hourly records. Unavailable vital-sign measurements may be left blank. "
        "Context variables (Age, Gender, HospAdmTime) remain fixed across observations."
    )

    hours = list(range(int(start_iculos), int(start_iculos) + 6))
    initial_data = pd.DataFrame({
        "ICULOS": hours,
        "HR": [82.0, 85.0, 88.0, 92.0, 96.0, 100.0],
        "O2Sat": [98.0, 98.0, 97.0, 96.0, 95.0, 94.0],
        "Temp": [37.0, 37.1, 37.2, 37.4, 37.5, 37.7],
        "SBP": [120.0, 118.0, 116.0, 112.0, 108.0, 104.0],
        "MAP": [85.0, 84.0, 82.0, 80.0, 77.0, 74.0],
        "Resp": [18.0, 19.0, 20.0, 21.0, 23.0, 24.0]
    })

    edited_data = st.data_editor(
        initial_data,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        disabled=["ICULOS"],
        column_config={
            "ICULOS": st.column_config.NumberColumn("ICU Hour", format="%d"),
            "HR": st.column_config.NumberColumn("Heart Rate (bpm)"),
            "O2Sat": st.column_config.NumberColumn("Oxygen Saturation (%)"),
            "Temp": st.column_config.NumberColumn("Temperature (°C)"),
            "SBP": st.column_config.NumberColumn("Systolic BP (mmHg)"),
            "MAP": st.column_config.NumberColumn("Mean Arterial Pressure (mmHg)"),
            "Resp": st.column_config.NumberColumn("Respiratory Rate (breaths/min)")
        }
    )

    if st.button("Run Assessment", type="primary"):
        manual_data = edited_data.copy()
        manual_data["Age"] = age
        manual_data["Gender"] = 0 if gender_label == "Female" else 1
        manual_data["HospAdmTime"] = hospital_admission_time

        try:
            execute_assessment(manual_data)
        except Exception as error:
            st.error(f"Assessment could not be completed: {error}")


# =====================================================
# 2. CSV UPLOAD
# =====================================================
elif input_method == "CSV Upload":
    st.subheader("Upload Patient Observation CSV")
    st.write(
        "Upload a structured CSV file containing consecutive hourly observations for a single patient."
    )
    st.caption(
        "Required columns: `HR, O2Sat, Temp, SBP, MAP, Resp, Age, Gender, ICULOS, HospAdmTime`. "
        "At least 6 consecutive hourly records are required."
    )

    # Provide CSV Template Download
    if TEMPLATE_PATH.exists():
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            template_bytes = f.read()
        st.download_button(
            label="Download Sample CSV Template",
            data=template_bytes,
            file_name="sample_patient_template.csv",
            mime="text/csv",
            help="Download a pre-formatted template with 6 consecutive hours."
        )

    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            csv_data = pd.read_csv(uploaded_file)
            st.write("### Uploaded Data Preview")
            st.dataframe(csv_data.head(10), use_container_width=True)
            st.write(f"Total observations in file: **{len(csv_data)}**")

            if st.button("Run Assessment on Uploaded Data", type="primary"):
                execute_assessment(csv_data)
        except Exception as error:
            st.error(f"Unable to process CSV file: {error}")


# =====================================================
# 3. DEMO MODE
# =====================================================
else:
    st.subheader("Synthetic Clinical Demonstration Scenarios")
    st.write(
        "Explore model responses on synthetic patient trajectories illustrating "
        "different physiological patterns. None of these records contain real patient information."
    )

    demo_scenario = st.selectbox(
        "Select Clinical Scenario",
        [
            "Scenario 1: Progressive Deterioration (Tachycardia, Hypotension & Fever)",
            "Scenario 2: Stable ICU Trajectory (Normotensive, Euthermic)",
            "Scenario 3: Respiratory Decline with Intermittent Missing Measurements"
        ]
    )

    if "Scenario 1" in demo_scenario:
        demo_df = pd.DataFrame({
            "ICULOS": [10, 11, 12, 13, 14, 15],
            "HR": [82, 86, 92, 98, 104, 112],
            "O2Sat": [98, 97, 96, 95, 94, 92],
            "Temp": [37.0, 37.2, 37.5, 37.8, 38.1, 38.4],
            "SBP": [122, 118, 112, 106, 100, 94],
            "MAP": [86, 82, 78, 74, 70, 66],
            "Resp": [18, 19, 21, 23, 26, 28],
            "Age": [68] * 6,
            "Gender": [1] * 6,
            "HospAdmTime": [-8.0] * 6
        })
        scenario_note = "Trajectory shows rising heart rate, increasing core temperature, progressive hypotension, and tachypnea."

    elif "Scenario 2" in demo_scenario:
        demo_df = pd.DataFrame({
            "ICULOS": [1, 2, 3, 4, 5, 6],
            "HR": [72, 74, 73, 75, 74, 72],
            "O2Sat": [99, 98, 99, 98, 98, 99],
            "Temp": [36.8, 36.9, 36.8, 37.0, 36.9, 36.8],
            "SBP": [124, 122, 120, 122, 124, 121],
            "MAP": [88, 86, 85, 87, 86, 85],
            "Resp": [16, 16, 17, 16, 17, 16],
            "Age": [55] * 6,
            "Gender": [0] * 6,
            "HospAdmTime": [-3.0] * 6
        })
        scenario_note = "Trajectory shows stable physiological vital signs within expected normal limits."

    else:
        demo_df = pd.DataFrame({
            "ICULOS": [14, 15, 16, 17, 18, 19],
            "HR": [80, 84, np.nan, 90, 94, 98],
            "O2Sat": [97, 96, 95, 94, 93, 91],
            "Temp": [37.1, 37.1, 37.3, np.nan, 37.4, 37.6],
            "SBP": [118, 116, 114, 112, 110, 108],
            "MAP": [84, 82, 80, 78, 77, 75],
            "Resp": [20, 22, 24, 25, 27, 29],
            "Age": [72] * 6,
            "Gender": [1] * 6,
            "HospAdmTime": [-14.0] * 6
        })
        scenario_note = "Trajectory demonstrates acute respiratory rate elevation with declining oxygen saturation and intermittent unmeasured values."

    st.info(f"**Scenario Profile:** {scenario_note}")
    st.dataframe(demo_df, use_container_width=True, hide_index=True)

    if st.button("Evaluate Selected Demo Scenario", type="primary"):
        try:
            execute_assessment(demo_df)
        except Exception as error:
            st.error(f"Demo assessment could not be completed: {error}")


# -------------------------------------------------
# RESULTS NAVIGATION
# -------------------------------------------------
if "assessment_results" in st.session_state:
    st.divider()
    st.subheader("Assessment Completed")
    st.write(
        "Assessment outputs have been cached. Review detailed comparisons and threshold "
        "analyses in **Prediction Results**, or explore feature attributions in **Explainability**."
    )
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        st.page_link(
            "pages/3_Prediction_Results.py",
            label="Open Prediction Results",
            icon=":material/analytics:"
        )
    with col_nav2:
        st.page_link(
            "pages/4_Explainability.py",
            label="Open Explainability & SHAP",
            icon=":material/psychology:"
        )
