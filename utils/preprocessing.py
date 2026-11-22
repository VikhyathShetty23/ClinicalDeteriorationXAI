
import numpy as np
import pandas as pd


# --------------------------------------------------
# FEATURE DEFINITIONS
# --------------------------------------------------

VITAL_FEATURES = [
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "Resp",
]

CONTEXT_FEATURES = [
    "Age",
    "Gender",
    "ICULOS",
    "HospAdmTime",
]


# --------------------------------------------------
# STEP 1: VALIDATE INPUT DATA
# --------------------------------------------------

def validate_observations(df):
    required_columns = VITAL_FEATURES + CONTEXT_FEATURES

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if len(df) < 6:
        raise ValueError(
            "At least 6 hourly observations are required."
        )

    df = df.copy()

    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("ICULOS").reset_index(drop=True)

    if df["ICULOS"].isna().any():
        raise ValueError("ICULOS cannot be missing.")

    if df["ICULOS"].duplicated().any():
        raise ValueError("Duplicate ICU observation hours found.")

    if not np.all(np.diff(df["ICULOS"].to_numpy()) == 1):
        raise ValueError(
        "ICU observations must be consecutive hourly records. "
        "Missing or skipped ICU hours were detected."
    )  

    if df[CONTEXT_FEATURES].isna().any().any():
        raise ValueError(
            "Age, Gender, ICULOS and HospAdmTime must have valid values."
        )

    # Check that observations are consecutive hourly records
    icu_hours = df["ICULOS"].to_numpy()

    if not np.all(np.diff(icu_hours) == 1):
        raise ValueError(
            "Observations must cover consecutive ICU hours "
            "without gaps."
        )

    # Ensure one patient context is used throughout the sequence
    for col in ["Age", "Gender", "HospAdmTime"]:
        if df[col].nunique() != 1:
            raise ValueError(
                f"{col} must remain the same across all observations."
            )

    return df


# --------------------------------------------------
# STEP 2: MISSINGNESS INDICATORS AND FORWARD FILL
# --------------------------------------------------

def handle_missing_values(df):
    """
    Create missingness indicators before forward filling.
    Forward fill uses previous observations only.
    """

    df = df.copy()

    for col in VITAL_FEATURES:
        df[col + "_missing"] = (
            df[col].isna().astype(int)
        )

    df[VITAL_FEATURES] = (
        df[VITAL_FEATURES].ffill()
    )

    return df


# --------------------------------------------------
# STEP 3: CREATE TEMPORAL FEATURES
# --------------------------------------------------

def create_temporal_features(df):
    """
    Reproduce the temporal feature engineering
    used during model training.
    """

    df = df.copy()

    temporal_features = []

    for col in VITAL_FEATURES:

        # Rolling averages and standard deviations
        for window in [3, 6]:

            mean_name = f"{col}_mean_{window}h"
            std_name = f"{col}_std_{window}h"

            df[mean_name] = (
                df[col]
                .rolling(window, min_periods=1)
                .mean()
            )

            df[std_name] = (
                df[col]
                .rolling(window, min_periods=2)
                .std()
            )

            temporal_features.extend([
                mean_name,
                std_name
            ])

        # Change from previous observation
        delta_name = f"{col}_change_1h"

        df[delta_name] = df[col].diff()

        temporal_features.append(delta_name)

    # Shock Index
    df["ShockIndex"] = (
        df["HR"] /
        df["SBP"].replace(0, np.nan)
    )

    # Modified Shock Index
    df["ModifiedShockIndex"] = (
        df["HR"] /
        df["MAP"].replace(0, np.nan)
    )

    temporal_features.extend([
        "ShockIndex",
        "ModifiedShockIndex"
    ])

    return df


# --------------------------------------------------
# STEP 4: PREPARE MODEL INPUTS
# --------------------------------------------------

def prepare_model_inputs(
    observations,
    config,
    xgb_imputer,
    clinical_imputer,
    lstm_imputer,
    lstm_scaler
):
    """
    Prepare input arrays for all three trained models.

    Returns:
        original_xgb_input
        clinical_xgb_input
        lstm_input
        processed_data
    """

    # Validate input
    data = validate_observations(observations)

    # Missingness indicators and forward filling
    data = handle_missing_values(data)

    # Temporal feature engineering
    data = create_temporal_features(data)

    # Replace infinite values
    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Retrieve exact training feature order
    xgb_features = config["xgb_features"]
    clinical_features = config["clinical_features"]
    lstm_features = config["lstm_features"]

    # -------------------------------
    # ORIGINAL XGBOOST
    # -------------------------------

    X_xgb = data[xgb_features].copy()

    X_xgb = xgb_imputer.transform(X_xgb)

    # Use the latest hourly observation
    original_xgb_input = X_xgb[-1:].astype(
        np.float32
    )

    # -------------------------------
    # CLINICAL-ONLY XGBOOST
    # -------------------------------

    X_clinical = data[clinical_features].copy()

    X_clinical = clinical_imputer.transform(
        X_clinical
    )

    clinical_xgb_input = X_clinical[-1:].astype(
        np.float32
    )

    # -------------------------------
    # LSTM
    # -------------------------------

    X_lstm = data[lstm_features].copy()

    X_lstm = lstm_imputer.transform(X_lstm)

    X_lstm = lstm_scaler.transform(
        X_lstm
    ).astype(np.float32)

    sequence_length = config["sequence_length"]

    if len(X_lstm) < sequence_length:
        raise ValueError(
            f"LSTM requires {sequence_length} observations."
        )

    # Latest six consecutive observations
    lstm_input = X_lstm[
        -sequence_length:
    ][np.newaxis, :, :]

    return {
        "original_xgb": original_xgb_input,
        "clinical_xgb": clinical_xgb_input,
        "lstm": lstm_input,
        "processed_data": data,
    }
