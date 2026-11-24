
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

SCORE_COLUMN = "original_xgboost_score"


# ==================================================
# 2. LOAD DATA
# ==================================================

print("Loading full-test predictions...")

df = pd.read_csv(PREDICTIONS_PATH)

df["PatientID"] = df["PatientID"].astype(str)
df["ICULOS"] = pd.to_numeric(df["ICULOS"], errors="raise")

if df.duplicated(["PatientID", "ICULOS"]).any():
    raise ValueError("Duplicate patient-hour records found.")

df["Alert"] = (
    df[SCORE_COLUMN] >= THRESHOLD
).astype(int)

print("Total records:", len(df))
print("Total patients:", df["PatientID"].nunique())


# ==================================================
# 3. IDENTIFY SEPSIS AND NON-SEPSIS PATIENTS
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

print("Sepsis patients:", len(sepsis_ids))
print("Non-sepsis patients:", len(non_sepsis_ids))


# ==================================================
# 4. FUNCTION TO IDENTIFY ALERT EPISODES
# ==================================================

def extract_episodes(data, group_name):

    episodes = []

    for patient_id, patient_df in data.groupby("PatientID"):

        patient_df = patient_df.sort_values("ICULOS").copy()

        # An episode continues only when:
        # 1. The current record is an alert.
        # 2. The ICU hour is consecutive.
        # A missing hour or a non-alert ends the episode.

        previous_alert = patient_df["Alert"].shift(
            fill_value=0
        )

        previous_hour = patient_df["ICULOS"].shift()

        consecutive_hour = (
            patient_df["ICULOS"] - previous_hour == 1
        )

        new_episode = (
            (patient_df["Alert"] == 1)
            & (
                (previous_alert == 0)
                | (~consecutive_hour)
            )
        )

        patient_df["EpisodeID"] = new_episode.cumsum()

        alert_rows = patient_df[
            patient_df["Alert"] == 1
        ]

        if alert_rows.empty:
            continue

        for episode_id, episode_df in alert_rows.groupby(
            "EpisodeID"
        ):

            start_hour = int(episode_df["ICULOS"].min())
            end_hour = int(episode_df["ICULOS"].max())

            duration = len(episode_df)

            episodes.append({
                "Group": group_name,
                "PatientID": patient_id,
                "EpisodeID": int(episode_id),
                "Start_ICULOS": start_hour,
                "End_ICULOS": end_hour,
                "Duration_Hours": duration,
                "Alert_Count": duration,
                "Is_Isolated_Alert": int(duration == 1),
                "Is_Sustained_Alert": int(duration >= 2)
            })

    return pd.DataFrame(episodes)


# ==================================================
# 5. SEPSIS ALERT EPISODES
# ==================================================

print("\nAnalysing sepsis alert episodes...")

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

# Restrict to the 2-6 hour pre-onset window
sepsis_window = sepsis_df[
    (sepsis_df["LeadTimeHours"] >= 2)
    & (sepsis_df["LeadTimeHours"] <= 6)
].copy()

sepsis_episodes = extract_episodes(
    sepsis_window,
    "Sepsis"
)


# ==================================================
# 6. NON-SEPSIS FALSE-ALERT EPISODES
# ==================================================

print("Analysing non-sepsis alert episodes...")

non_sepsis_df = df[
    df["PatientID"].isin(non_sepsis_ids)
].copy()

non_sepsis_episodes = extract_episodes(
    non_sepsis_df,
    "Non-Sepsis"
)


# ==================================================
# 7. COMBINE EPISODES
# ==================================================

all_episodes = pd.concat(
    [sepsis_episodes, non_sepsis_episodes],
    ignore_index=True
)

if all_episodes.empty:
    raise ValueError("No alert episodes were identified.")

all_episodes.to_csv(
    OUTPUT_DIR / "alert_episode_details.csv",
    index=False
)


# ==================================================
# 8. SUMMARIZE EPISODES
# ==================================================

def summarize_episodes(
    episodes,
    group_name,
    observed_hours,
    total_patients
):

    if episodes.empty:
        return {
            "Group": group_name,
            "Patients": total_patients,
            "Observed_Hours": observed_hours,
            "Patients_With_Alerts": 0,
            "Total_Episodes": 0,
            "Isolated_Episodes": 0,
            "Sustained_Episodes": 0,
            "Mean_Duration_Hours": np.nan,
            "Median_Duration_Hours": np.nan,
            "Episodes_Per_24_Patient_Hours": 0
        }

    total_episodes = len(episodes)

    isolated = int(
        episodes["Is_Isolated_Alert"].sum()
    )

    sustained = int(
        episodes["Is_Sustained_Alert"].sum()
    )

    patients_with_alerts = (
        episodes["PatientID"].nunique()
    )

    episode_rate = (
        total_episodes / observed_hours * 24
        if observed_hours else np.nan
    )

    return {
        "Group": group_name,
        "Patients": total_patients,
        "Observed_Hours": observed_hours,
        "Patients_With_Alerts": patients_with_alerts,
        "Total_Episodes": total_episodes,
        "Isolated_Episodes": isolated,
        "Sustained_Episodes": sustained,
        "Mean_Duration_Hours": episodes[
            "Duration_Hours"
        ].mean(),
        "Median_Duration_Hours": episodes[
            "Duration_Hours"
        ].median(),
        "Episodes_Per_24_Patient_Hours": episode_rate
    }


sepsis_observed_hours = len(sepsis_window)
non_sepsis_observed_hours = len(non_sepsis_df)

sepsis_summary = summarize_episodes(
    sepsis_episodes,
    "Sepsis (2-6h pre-onset)",
    sepsis_observed_hours,
    len(sepsis_ids)
)

non_sepsis_summary = summarize_episodes(
    non_sepsis_episodes,
    "Non-Sepsis (full observation)",
    non_sepsis_observed_hours,
    len(non_sepsis_ids)
)

summary_df = pd.DataFrame([
    sepsis_summary,
    non_sepsis_summary
])

summary_df.to_csv(
    OUTPUT_DIR / "alert_episode_summary.csv",
    index=False
)


# ==================================================
# 9. DISPLAY RESULTS
# ==================================================

print("\n" + "=" * 65)
print("ALERT EPISODE ANALYSIS")
print("=" * 65)

print("\nSEPSIS ALERT EPISODES")
print("Episodes:", len(sepsis_episodes))

if not sepsis_episodes.empty:
    print(
        "Patients with episodes:",
        sepsis_episodes["PatientID"].nunique()
    )
    print(
        "Isolated episodes:",
        int(sepsis_episodes["Is_Isolated_Alert"].sum())
    )
    print(
        "Sustained episodes:",
        int(sepsis_episodes["Is_Sustained_Alert"].sum())
    )
    print(
        "Mean duration:",
        round(sepsis_episodes["Duration_Hours"].mean(), 2)
    )

print("\nNON-SEPSIS FALSE-ALERT EPISODES")
print("Episodes:", len(non_sepsis_episodes))

if not non_sepsis_episodes.empty:
    print(
        "Patients with false alerts:",
        non_sepsis_episodes["PatientID"].nunique()
    )
    print(
        "Isolated episodes:",
        int(non_sepsis_episodes["Is_Isolated_Alert"].sum())
    )
    print(
        "Sustained episodes:",
        int(non_sepsis_episodes["Is_Sustained_Alert"].sum())
    )
    print(
        "Mean duration:",
        round(non_sepsis_episodes["Duration_Hours"].mean(), 2)
    )

print("\nSUMMARY TABLE")
print(summary_df.to_string(index=False))

print("\nSaved files:")
print(OUTPUT_DIR / "alert_episode_details.csv")
print(OUTPUT_DIR / "alert_episode_summary.csv")

print("\nAlert episode analysis completed.")
