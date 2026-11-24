
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
LEAD_TIMES = [6, 4, 2]
TOTAL_SEPSIS_PATIENTS = 148

SCORE_COLUMN = "original_xgboost_score"


# ==================================================
# 2. LOAD FULL TEST PREDICTIONS
# ==================================================

print("Loading full test predictions...")

df = pd.read_csv(PREDICTIONS_PATH)

df["PatientID"] = df["PatientID"].astype(str)
df["ICULOS"] = pd.to_numeric(df["ICULOS"], errors="raise")

if df.duplicated(["PatientID", "ICULOS"]).any():
    raise ValueError("Duplicate patient-hour records detected.")

print("Prediction rows:", len(df))
print("Unique patients:", df["PatientID"].nunique())


# ==================================================
# 3. IDENTIFY FIRST RECORDED SEPSIS ONSET
# ==================================================

positive_rows = df[df["SepsisLabel"] == 1].copy()

onsets = (
    positive_rows.groupby("PatientID")["ICULOS"]
    .min()
    .reset_index()
    .rename(columns={"ICULOS": "SepsisOnsetHour"})
)

print("\nSepsis patients:", len(onsets))

if len(onsets) != TOTAL_SEPSIS_PATIENTS:
    raise ValueError(
        f"Expected {TOTAL_SEPSIS_PATIENTS} sepsis patients, "
        f"found {len(onsets)}."
    )


# ==================================================
# 4. MATCH EACH PATIENT WITH THEIR ONSET
# ==================================================

sepsis_predictions = df.merge(
    onsets,
    on="PatientID",
    how="inner",
    validate="many_to_one"
)


# ==================================================
# 5. EVALUATE EXACT LEAD TIMES
# ==================================================

summary = []
patient_details = []

for lead in LEAD_TIMES:

    print(f"\nEvaluating {lead} hours before onset...")

    target_hour = (
        sepsis_predictions["SepsisOnsetHour"] - lead
    )

    selected = sepsis_predictions[
        sepsis_predictions["ICULOS"] == target_hour
    ].copy()

    selected["Detected"] = (
        selected[SCORE_COLUMN] >= THRESHOLD
    ).astype(int)

    eligible = selected["PatientID"].nunique()
    detected = int(selected["Detected"].sum())
    missed = eligible - detected
    unavailable = TOTAL_SEPSIS_PATIENTS - eligible

    eligible_detection_rate = (
        detected / eligible * 100 if eligible else np.nan
    )

    full_cohort_detection_rate = (
        detected / TOTAL_SEPSIS_PATIENTS * 100
    )

    summary.append({
        "Lead_Time_Hours": lead,
        "Total_Sepsis_Patients": TOTAL_SEPSIS_PATIENTS,
        "Eligible_Patients": eligible,
        "Unavailable_Patients": unavailable,
        "Detected_Patients": detected,
        "Missed_Eligible_Patients": missed,
        "Detection_Rate_Among_Eligible_Pct":
            eligible_detection_rate,
        "Detection_Rate_Among_All_Sepsis_Patients_Pct":
            full_cohort_detection_rate,
        "Threshold": THRESHOLD
    })

    selected["Lead_Time_Hours"] = lead

    patient_details.append(
        selected[[
            "PatientID",
            "SepsisOnsetHour",
            "ICULOS",
            "Lead_Time_Hours",
            SCORE_COLUMN,
            "Detected"
        ]]
    )

    print("Eligible patients:", eligible)
    print("Unavailable patients:", unavailable)
    print("Detected:", detected)
    print("Missed among eligible:", missed)
    print(
        "Detection rate among eligible:",
        round(eligible_detection_rate, 2), "%"
    )
    print(
        "Detection rate among all 148 patients:",
        round(full_cohort_detection_rate, 2), "%"
    )


# ==================================================
# 6. SAVE RESULTS
# ==================================================

summary_df = pd.DataFrame(summary)

details_df = pd.concat(
    patient_details,
    ignore_index=True
)

summary_df.to_csv(
    OUTPUT_DIR / "lead_time_summary_full_test.csv",
    index=False
)

details_df.to_csv(
    OUTPUT_DIR / "lead_time_patient_details_full_test.csv",
    index=False
)

print("\n" + "=" * 65)
print("FULL-TEST LEAD-TIME EVALUATION COMPLETED")
print("=" * 65)

print(
    summary_df.to_string(index=False)
)

print("\nSaved:")
print(OUTPUT_DIR / "lead_time_summary_full_test.csv")
print(OUTPUT_DIR / "lead_time_patient_details_full_test.csv")
