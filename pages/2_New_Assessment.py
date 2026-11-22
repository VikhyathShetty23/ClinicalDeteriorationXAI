
import streamlit as st
import pandas as pd
import numpy as np

from utils.preprocessing import prepare_model_inputs
from utils.prediction import predict_all


st.set_page_config(
    page_title="New Assessment | Clinical XAI",
    layout="wide"
)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
CSS_PATH = BASE_DIR / "assets" / "style.css"

with open(CSS_PATH, "r", encoding="utf-8") as file:
    st.markdown(
        f"<style>{file.read()}</style>",
        unsafe_allow_html=True
    )

# -----------------------------
# Sidebar
# -----------------------------
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


# -----------------------------
# Page heading
# -----------------------------
st.title("New Patient Assessment")

st.write(
    "Enter hourly patient observations manually or upload a CSV file. "
    "The latest observation will be evaluated by the three trained models."
)

st.info(
    "Research prototype only. Model outputs are not clinical diagnoses "
    "or treatment recommendations."
)

models = st.session_state.get("models")

if models is None:
    st.error("Models are not loaded. Return to the Overview page.")
    st.stop()


# -----------------------------
# Input method
# -----------------------------
input_method = st.radio(
    "Choose input method",
    ["Manual Entry", "CSV Upload"],
    horizontal=True
)


# -----------------------------
# Shared prediction function
# -----------------------------
def run_assessment(observations):

    model_inputs = prepare_model_inputs(
        observations,
        models["config"],
        models["xgb_imputer"],
        models["clinical_imputer"],
        models["lstm_imputer"],
        models["lstm_scaler"]
    )

    results = predict_all(models, model_inputs)

    st.session_state["assessment_results"] = results
    st.session_state["assessment_observations"] = observations.copy()
    st.session_state["assessment_processed"] = (
        model_inputs["processed_data"].copy()
    )

    st.success("Prediction completed successfully.")

    st.write("### Prediction summary")

    col1, col2, col3 = st.columns(3)

    model_cards = [
        ("Original XGBoost", "original_xgb", col1),
        ("Clinical-only XGBoost", "clinical_xgb", col2),
        ("Temporal Attention LSTM", "lstm", col3)
    ]

    for label, key, column in model_cards:
        with column:
            probability = results["probabilities"][key]
            alert = results["alerts"][key]

            st.metric(
                label,
                f"{probability * 100:.2f}%"
            )

            if alert:
                st.warning("Model threshold crossed")
            else:
                st.success("Below model threshold")

    st.caption(
        "Probabilities are model outputs and should not be interpreted "
        "as calibrated clinical risk."
    )


# =====================================================
# MANUAL ENTRY
# =====================================================
if input_method == "Manual Entry":

    st.subheader("Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age (years)",
            min_value=18,
            max_value=120,
            value=65
        )

    with col2:
        gender_label = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

    with col3:
        hospital_admission_time = st.number_input(
            "HospAdmTime",
            value=-5.0,
            step=1.0,
            help="Hours between hospital admission and ICU admission."
        )

    st.subheader("Hourly Observation Details")

    start_iculos = st.number_input(
        "Starting ICU observation hour (ICULOS)",
        min_value=1,
        value=10,
        step=1
    )

    st.caption(
        "Enter six consecutive hourly observations. Leave unavailable "
        "vital-sign measurements blank; do not enter invented values."
    )

    hours = list(range(int(start_iculos), int(start_iculos) + 6))

    initial_data = pd.DataFrame({
        "ICULOS": hours,
        "HR": [np.nan] * 6,
        "O2Sat": [np.nan] * 6,
        "Temp": [np.nan] * 6,
        "SBP": [np.nan] * 6,
        "MAP": [np.nan] * 6,
        "Resp": [np.nan] * 6
    })

    edited_data = st.data_editor(
        initial_data,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        disabled=["ICULOS"],
        column_config={
            "ICULOS": st.column_config.NumberColumn(
                "ICU Hour",
                format="%d"
            ),
            "HR": st.column_config.NumberColumn(
                "Heart Rate",
                help="Beats per minute"
            ),
            "O2Sat": st.column_config.NumberColumn(
                "Oxygen Saturation",
                help="Percentage"
            ),
            "Temp": st.column_config.NumberColumn(
                "Temperature",
                help="Degrees Celsius"
            ),
            "SBP": st.column_config.NumberColumn(
                "Systolic BP",
                help="mmHg"
            ),
            "MAP": st.column_config.NumberColumn(
                "Mean Arterial Pressure",
                help="mmHg"
            ),
            "Resp": st.column_config.NumberColumn(
                "Respiratory Rate",
                help="Breaths per minute"
            )
        }
    )

    if st.button("Run Assessment", type="primary"):

        manual_data = edited_data.copy()

        manual_data["Age"] = age

        # PhysioNet Gender encoding:
        # 0 = Female, 1 = Male
        manual_data["Gender"] = (
            0 if gender_label == "Female" else 1
        )

        manual_data["HospAdmTime"] = hospital_admission_time

        try:
            run_assessment(manual_data)

        except Exception as error:
            st.error(f"Assessment could not be completed: {error}")


# =====================================================
# CSV UPLOAD
# =====================================================
else:

    st.subheader("Upload Patient Observation CSV")

    st.write(
        "Upload a CSV containing hourly observations for one patient."
    )

    st.caption(
        "Required columns: HR, O2Sat, Temp, SBP, MAP, Resp, "
        "Age, Gender, ICULOS and HospAdmTime."
    )

    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            csv_data = pd.read_csv(uploaded_file)

            st.write("### Uploaded data preview")
            st.dataframe(
                csv_data.head(10),
                use_container_width=True
            )

            st.write(
                f"Number of uploaded observations: {len(csv_data)}"
            )

            if st.button("Run Assessment", type="primary"):

                run_assessment(csv_data)

        except Exception as error:
            st.error(f"Unable to read CSV: {error}")


# -----------------------------
# Results navigation
# -----------------------------
if "assessment_results" in st.session_state:

    st.divider()

    st.subheader("Assessment completed")

    st.write(
        "Your latest assessment results are available in the "
        "Prediction Results section."
    )

    st.page_link(
        "pages/3_Prediction_Results.py",
        label="Open Prediction Results",
        icon=":material/analytics:"
    )
