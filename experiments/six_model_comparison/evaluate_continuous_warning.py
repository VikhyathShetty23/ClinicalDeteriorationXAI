
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

THRESHOLD = 0.748707

MIN_LEAD_HOURS = 2
MAX_LEAD_HOURS = 6

SCORE_COLUMN = "original_xgboost_score"


# ==================================================
# 2. LOAD PREDICTIONS
# ==================================================

print("Loading full-test predictions...")

df = pd.read_csv(PREDICTIONS_PATH)

df["PatientID"] = df["PatientID"].astype(str)
df["ICULOS"] = pd.to_numeric(df["ICULOS"], errors="raise")
df["SepsisLabel"] = df["SepsisLabel"].astype(int)

if df.duplicated(["PatientID", "ICULOS"]).any():
    raise ValueError("Duplicate patient-hour records detected.")

if df[SCORE_COLUMN].isna().any():
    raise ValueError("Missing prediction scores detected.")

print("Total records:", len(df))
print("Total patients:", df["PatientID"].nunique())


# ==================================================
# 3. IDENTIFY FIRST SEPSIS ONSET
# ==================================================

positive_rows = df[df["SepsisLabel"] == 1]

onsets = (
    positive_rows.groupby("PatientID")["ICULOS"]
    .min()
    .reset_index()
    .rename(columns={"ICULOS": "SepsisOnsetHour"})
)

sepsis_patients = set(onsets["PatientID"])

all_patients = set(df["PatientID"])

non_sepsis_patients = all_patients - sepsis_patients

print("\nSepsis patients:", len(sepsis_patients))
print("Non-sepsis patients:", len(non_sepsis_patients))


# ==================================================
# 4. CONTINUOUS WARNING WINDOW
# ==================================================

print("\nEvaluating continuous warning window...")
print(
    f"Window: {MIN_LEAD_HOURS} to "
    f"{MAX_LEAD_HOURS} hours before onset"
)

sepsis_df = df[
    df["PatientID"].isin(sepsis_patients)
].merge(
    onsets,
    on="PatientID",
    how="left",
    validate="many_to_one"
)

# Positive lead means prediction occurred before onset
sepsis_df["LeadTimeHours"] = (
    sepsis_df["SepsisOnsetHour"] - sepsis_df["ICULOS"]
)

# Select observations from 2 through 6 hours before onset
window_df = sepsis_df[
    (sepsis_df["LeadTimeHours"] >= MIN_LEAD_HOURS)
    & (sepsis_df["LeadTimeHours"] <= MAX_LEAD_HOURS)
].copy()

window_df["Alert"] = (
    window_df[SCORE_COLUMN] >= THRESHOLD
).astype(int)


# ==================================================
# 5. PATIENT-LEVEL WARNING RESULTS
# ==================================================

patient_results = []

for patient_id in sorted(sepsis_patients):

    patient_window = window_df[
        window_df["PatientID"] == patient_id
    ]

    available_hours = len(patient_window)

    alerts = patient_window[
        patient_window["Alert"] == 1
    ]

    detected = len(alerts) > 0

    if detected:
        # Largest lead time = earliest alert before onset
        earliest_warning = float(
            alerts["LeadTimeHours"].max()
        )

        first_alert_score = float(
            alerts.loc[
                alerts["LeadTimeHours"].idxmax(),
                SCORE_COLUMN
            ]
        )
    else:
        earliest_warning = np.nan
        first_alert_score = np.nan

    patient_results.append({
        "PatientID": patient_id,
        "SepsisOnsetHour": float(
            onsets.loc[
                onsets["PatientID"] == patient_id,
                "SepsisOnsetHour"
            ].iloc[0]
        ),
        "Available_Window_Hours": available_hours,
        "Detected_In_Window": int(detected),
        "Earliest_Warning_Hours": earliest_warning,
        "Score_At_Earliest_Warning": first_alert_score,
        "Alert_Hours_In_Window": len(alerts)
    })


patient_results_df = pd.DataFrame(patient_results)

total_sepsis = len(patient_results_df)

eligible = int(
    (patient_results_df["Available_Window_Hours"] > 0).sum()
)

detected = int(
    patient_results_df["Detected_In_Window"].sum()
)

no_warning = total_sepsis - detected

unavailable = total_sepsis - eligible

detection_all = detected / total_sepsis * 100

detection_eligible = (
    detected / eligible * 100 if eligible else np.nan
)

detected_patients = patient_results_df[
    patient_results_df["Detected_In_Window"] == 1
]


# ==================================================
# 6. WARNING-TIME DISTRIBUTION
# ==================================================

if len(detected_patients) > 0:

    warning_times = detected_patients[
        "Earliest_Warning_Hours"
    ]

    mean_warning = float(warning_times.mean())
    median_warning = float(warning_times.median())

    warning_6h = int((warning_times >= 6).sum())
    warning_4_6h = int(
        ((warning_times >= 4) & (warning_times < 6)).sum()
    )
    warning_2_4h = int(
        ((warning_times >= 2) & (warning_times < 4)).sum()
    )

else:
    mean_warning = np.nan
    median_warning = np.nan
    warning_6h = 0
    warning_4_6h = 0
    warning_2_4h = 0


# ==================================================
# 7. FALSE-ALERT BURDEN IN NON-SEPSIS PATIENTS
# ==================================================

print("\nEvaluating non-sepsis patients...")

non_sepsis_df = df[
    df["PatientID"].isin(non_sepsis_patients)
].copy()

non_sepsis_df["Alert"] = (
    non_sepsis_df[SCORE_COLUMN] >= THRESHOLD
).astype(int)

patient_alerts = (
    non_sepsis_df.groupby("PatientID")
    .agg(
        Observed_Hours=("ICULOS", "count"),
        Alert_Hours=("Alert", "sum")
    )
    .reset_index()
)

patient_alerts["Any_False_Alert"] = (
    patient_alerts["Alert_Hours"] > 0
).astype(int)

patients_with_alert = int(
    patient_alerts["Any_False_Alert"].sum()
)

total_non_sepsis = len(non_sepsis_patients)

false_alert_patient_rate = (
    patients_with_alert / total_non_sepsis * 100
)

total_alert_hours = int(
    patient_alerts["Alert_Hours"].sum()
)

total_observed_hours = int(
    patient_alerts["Observed_Hours"].sum()
)

false_alert_hours_per_1000 = (
    total_alert_hours / total_observed_hours * 1000
    if total_observed_hours else np.nan
)


# ==================================================
# 8. SAVE OUTPUTS
# ==================================================

summary = {
    "Threshold": THRESHOLD,
    "Warning_Window_Hours": "2_to_6",
    "Total_Sepsis_Patients": total_sepsis,
    "Patients_With_Window_Observations": eligible,
    "Patients_Without_Window_Observations": unavailable,
    "Patients_Detected_In_Window": detected,
    "Patients_Not_Detected_In_Window": no_warning,
    "Detection_Rate_All_Sepsis_Pct": detection_all,
    "Detection_Rate_Eligible_Pct": detection_eligible,
    "Mean_Earliest_Warning_Hours": mean_warning,
    "Median_Earliest_Warning_Hours": median_warning,
    "Warnings_At_6_Hours": warning_6h,
    "Warnings_4_to_6_Hours": warning_4_6h,
    "Warnings_2_to_4_Hours": warning_2_4h,
    "Total_Non_Sepsis_Patients": total_non_sepsis,
    "Non_Sepsis_Patients_With_Any_Alert": patients_with_alert,
    "Non_Sepsis_Patient_Alert_Rate_Pct": false_alert_patient_rate,
    "Non_Sepsis_Alert_Hours": total_alert_hours,
    "Non_Sepsis_Observed_Hours": total_observed_hours,
    "False_Alert_Hours_Per_1000_Observed_Hours":
        false_alert_hours_per_1000
}

summary_df = pd.DataFrame([summary])

summary_df.to_csv(
    OUTPUT_DIR / "continuous_warning_summary.csv",
    index=False
)

patient_results_df.to_csv(
    OUTPUT_DIR / "continuous_warning_sepsis_patients.csv",
    index=False
)

patient_alerts.to_csv(
    OUTPUT_DIR / "continuous_warning_non_sepsis_patients.csv",
    index=False
)


# ==================================================
# 9. DISPLAY RESULTS
# ==================================================

print("\n" + "=" * 65)
print("CONTINUOUS EARLY-WARNING EVALUATION")
print("=" * 65)

print("\nSEPSIS PATIENT RESULTS")
print("Total sepsis patients:", total_sepsis)
print("Eligible patients:", eligible)
print("Unavailable patients:", unavailable)
print("Detected in window:", detected)
print("Not detected:", no_warning)

print(
    "Detection rate (all sepsis patients):",
    round(detection_all, 2), "%"
)

print(
    "Detection rate (eligible patients):",
    round(detection_eligible, 2), "%"
)

print("\nEARLIEST WARNING TIME")
print("Mean:", round(mean_warning, 2), "hours")
print("Median:", round(median_warning, 2), "hours")
print("Warnings at 6 hours:", warning_6h)
print("Warnings from 4 to under 6 hours:", warning_4_6h)
print("Warnings from 2 to under 4 hours:", warning_2_4h)

print("\nNON-SEPSIS FALSE-ALERT BURDEN")
print("Non-sepsis patients:", total_non_sepsis)
print("Patients with any alert:", patients_with_alert)
print(
    "Patient-level alert rate:",
    round(false_alert_patient_rate, 2), "%"
)
print("Total alert-hours:", total_alert_hours)
print("Total observed hours:", total_observed_hours)
print(
    "Alert-hours per 1000 observed hours:",
    round(false_alert_hours_per_1000, 2)
)

print("\nSaved results:")
print(OUTPUT_DIR / "continuous_warning_summary.csv")
print(OUTPUT_DIR / "continuous_warning_sepsis_patients.csv")
print(OUTPUT_DIR / "continuous_warning_non_sepsis_patients.csv")

print("\nEvaluation completed successfully.")
