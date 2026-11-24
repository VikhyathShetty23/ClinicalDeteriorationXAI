
from pathlib import Path
import pandas as pd
import numpy as np


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

PREDICTIONS_PATH = (
    BASE_DIR
    / "lead_time"
    / "original_xgboost_full_test_predictions.csv"
)

OUTPUT_DIR = BASE_DIR / "lead_time"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCORE_COLUMN = "original_xgboost_score"

THRESHOLDS = [
    0.65,
    0.70,
    0.748707,
    0.80,
    0.85
]

MIN_LEAD = 2
MAX_LEAD = 6


# ==================================================
# 2. LOAD DATA
# ==================================================

print("Loading full-test predictions...")

df = pd.read_csv(PREDICTIONS_PATH)

df["PatientID"] = df["PatientID"].astype(str)
df["ICULOS"] = pd.to_numeric(df["ICULOS"], errors="raise")

if df.duplicated(["PatientID", "ICULOS"]).any():
    raise ValueError("Duplicate patient-hour records found.")

if df[SCORE_COLUMN].isna().any():
    raise ValueError("Missing prediction scores found.")

print("Total records:", len(df))
print("Total patients:", df["PatientID"].nunique())


# ==================================================
# 3. IDENTIFY SEPSIS ONSET
# ==================================================

onsets = (
    df[df["SepsisLabel"] == 1]
    .groupby("PatientID")["ICULOS"]
    .min()
    .reset_index()
    .rename(columns={"ICULOS": "SepsisOnsetHour"})
)

sepsis_ids = set(onsets["PatientID"])
non_sepsis_ids = set(df["PatientID"]) - sepsis_ids

total_sepsis = len(sepsis_ids)
total_non_sepsis = len(non_sepsis_ids)

print("Sepsis patients:", total_sepsis)
print("Non-sepsis patients:", total_non_sepsis)


# ==================================================
# 4. PREPARE SEPSIS WARNING WINDOW
# ==================================================

sepsis_df = df[
    df["PatientID"].isin(sepsis_ids)
].merge(
    onsets,
    on="PatientID",
    how="left",
    validate="many_to_one"
)

sepsis_df["LeadTimeHours"] = (
    sepsis_df["SepsisOnsetHour"] - sepsis_df["ICULOS"]
)

window_df = sepsis_df[
    (sepsis_df["LeadTimeHours"] >= MIN_LEAD)
    & (sepsis_df["LeadTimeHours"] <= MAX_LEAD)
].copy()

eligible_patients = window_df["PatientID"].nunique()

print("\nPatients with observations in warning window:",
      eligible_patients)


# ==================================================
# 5. PREPARE NON-SEPSIS OBSERVATIONS
# ==================================================

non_sepsis_df = df[
    df["PatientID"].isin(non_sepsis_ids)
].copy()

total_non_sepsis_hours = len(non_sepsis_df)


# ==================================================
# 6. THRESHOLD SENSITIVITY ANALYSIS
# ==================================================

results = []

for threshold in THRESHOLDS:

    print("\nEvaluating threshold:", threshold)

    # ----------------------------------------------
    # Sepsis detection
    # ----------------------------------------------

    sepsis_window = window_df.copy()

    sepsis_window["Alert"] = (
        sepsis_window[SCORE_COLUMN] >= threshold
    ).astype(int)

    detected_patients = (
        sepsis_window.groupby("PatientID")["Alert"]
        .max()
    )

    detected_count = int(detected_patients.sum())

    eligible_detected = (
        sepsis_window.loc[
            sepsis_window["Alert"] == 1,
            "PatientID"
        ].nunique()
    )

    unavailable = total_sepsis - eligible_patients

    detection_all = (
        detected_count / total_sepsis * 100
    )

    detection_eligible = (
        detected_count / eligible_patients * 100
        if eligible_patients else np.nan
    )

    # ----------------------------------------------
    # Non-sepsis false-alert burden
    # ----------------------------------------------

    non_sepsis = non_sepsis_df.copy()

    non_sepsis["Alert"] = (
        non_sepsis[SCORE_COLUMN] >= threshold
    ).astype(int)

    patient_alerts = (
        non_sepsis.groupby("PatientID")["Alert"]
        .max()
    )

    patients_with_alert = int(patient_alerts.sum())

    patient_alert_rate = (
        patients_with_alert / total_non_sepsis * 100
    )

    false_alert_hours = int(non_sepsis["Alert"].sum())

    false_alert_rate = (
        false_alert_hours / total_non_sepsis_hours * 1000
    )

    # ----------------------------------------------
    # Store results
    # ----------------------------------------------

    results.append({
        "Threshold": threshold,

        "Total_Sepsis_Patients": total_sepsis,
        "Eligible_Sepsis_Patients": eligible_patients,
        "Unavailable_Sepsis_Patients": unavailable,
        "Detected_Sepsis_Patients": detected_count,

        "Detection_Rate_All_Sepsis_Pct": detection_all,
        "Detection_Rate_Eligible_Pct": detection_eligible,

        "Non_Sepsis_Patients": total_non_sepsis,
        "Non_Sepsis_Patients_With_Alerts": patients_with_alert,
        "Non_Sepsis_Patient_Alert_Rate_Pct":
            patient_alert_rate,

        "Non_Sepsis_Alert_Hours": false_alert_hours,
        "Non_Sepsis_Observed_Hours": total_non_sepsis_hours,
        "False_Alert_Hours_Per_1000_Hours":
            false_alert_rate
    })

    print("Detected sepsis patients:", detected_count)
    print("Detection rate:", round(detection_all, 2), "%")
    print("Non-sepsis patients with alerts:", patients_with_alert)
    print("Patient-level alert rate:",
          round(patient_alert_rate, 2), "%")
    print("Alert-hours per 1000 hours:",
          round(false_alert_rate, 2))


# ==================================================
# 7. SAVE RESULTS
# ==================================================

results_df = pd.DataFrame(results)

output_path = (
    OUTPUT_DIR / "threshold_sensitivity_results.csv"
)

results_df.to_csv(output_path, index=False)


# ==================================================
# 8. DISPLAY RESULTS
# ==================================================

print("\n" + "=" * 75)
print("THRESHOLD SENSITIVITY ANALYSIS")
print("=" * 75)

display_columns = [
    "Threshold",
    "Detected_Sepsis_Patients",
    "Detection_Rate_All_Sepsis_Pct",
    "Detection_Rate_Eligible_Pct",
    "Non_Sepsis_Patients_With_Alerts",
    "Non_Sepsis_Patient_Alert_Rate_Pct",
    "Non_Sepsis_Alert_Hours",
    "False_Alert_Hours_Per_1000_Hours"
]

print(
    results_df[display_columns].round(3).to_string(index=False)
)

print("\nResults saved to:")
print(output_path)

print("\nThreshold sensitivity analysis completed.")
