
import os
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

INPUT_FILE = os.path.join(
    RESULTS_DIR, "common_test_predictions.csv"
)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR, "paired_bootstrap_comparison.csv"
)

N_BOOTSTRAPS = 2000
RANDOM_STATE = 42

REFERENCE_MODEL = "Original XGBoost"

MODELS = {
    "Original XGBoost": "original_xgboost_score",
    "Random Forest": "random_forest_score",
    "Logistic Regression": "logistic_regression_score",
    "Linear SVM": "linear_svm_score",
    "Clinical Only XGBoost": "clinical_only_xgboost_score",
    "Temporal Attention LSTM": "temporal_attention_lstm_score"
}

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading common test predictions...")

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "PatientID",
    "SepsisLabel"
] + list(MODELS.values())

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df[required_columns].copy()

if df.isnull().any().any():
    raise ValueError("Missing values found in prediction data.")

y = df["SepsisLabel"].astype(int).to_numpy()

if set(np.unique(y)) != {0, 1}:
    raise ValueError("Both positive and negative labels are required.")

# --------------------------------------------------
# PATIENT GROUPS
# --------------------------------------------------

patient_codes, patient_ids = pd.factorize(
    df["PatientID"], sort=True
)

n_patients = len(patient_ids)
n_rows = len(df)

print(f"Test observations: {n_rows}")
print(f"Unique patients: {n_patients}")
print(f"Positive labels: {int(y.sum())}")

# Each row receives the bootstrap multiplicity
# of the patient to which it belongs.

# --------------------------------------------------
# MODEL SCORES
# --------------------------------------------------

scores = {
    name: df[column].to_numpy(dtype=float)
    for name, column in MODELS.items()
}

# --------------------------------------------------
# ORIGINAL TEST METRICS
# --------------------------------------------------

observed = {}

for name, model_scores in scores.items():

    observed[name] = {
        "AUROC": roc_auc_score(y, model_scores),
        "AUPRC": average_precision_score(y, model_scores)
    }

# --------------------------------------------------
# PAIRED PATIENT-LEVEL BOOTSTRAP
# --------------------------------------------------

rng = np.random.default_rng(RANDOM_STATE)

comparators = [
    name for name in MODELS
    if name != REFERENCE_MODEL
]

difference_samples = {
    name: {
        "AUROC": [],
        "AUPRC": []
    }
    for name in comparators
}

print("\nStarting paired patient-level bootstrap...")
print(f"Bootstrap repetitions: {N_BOOTSTRAPS}")

for iteration in range(N_BOOTSTRAPS):

    # Sample patients with replacement.
    sampled_patients = rng.integers(
        0, n_patients, size=n_patients
    )

    # Count how often each patient was selected.
    patient_counts = np.bincount(
        sampled_patients,
        minlength=n_patients
    )

    # Apply patient-level multiplicity to every
    # hourly observation belonging to that patient.
    sample_weights = patient_counts[patient_codes]

    # Skip a replicate if it contains only one class.
    active = sample_weights > 0

    y_boot = y[active]
    weights_boot = sample_weights[active]

    if len(np.unique(y_boot)) < 2:
        continue

    bootstrap_metrics = {}

    for name, model_scores in scores.items():

        scores_boot = model_scores[active]

        bootstrap_metrics[name] = {
            "AUROC": roc_auc_score(
                y_boot,
                scores_boot,
                sample_weight=weights_boot
            ),
            "AUPRC": average_precision_score(
                y_boot,
                scores_boot,
                sample_weight=weights_boot
            )
        }

    reference = bootstrap_metrics[REFERENCE_MODEL]

    # Same sampled patients are used for both models.
    for name in comparators:

        difference_samples[name]["AUROC"].append(
            reference["AUROC"] -
            bootstrap_metrics[name]["AUROC"]
        )

        difference_samples[name]["AUPRC"].append(
            reference["AUPRC"] -
            bootstrap_metrics[name]["AUPRC"]
        )

    if (iteration + 1) % 200 == 0:
        print(
            f"Completed {iteration + 1}/{N_BOOTSTRAPS}"
        )

# --------------------------------------------------
# SUMMARIZE PAIRED DIFFERENCES
# --------------------------------------------------

rows = []

for name in comparators:

    result = {
        "Comparison": f"{REFERENCE_MODEL} vs {name}"
    }

    for metric in ["AUROC", "AUPRC"]:

        differences = np.asarray(
            difference_samples[name][metric]
        )

        lower, upper = np.percentile(
            differences, [2.5, 97.5]
        )

        observed_difference = (
            observed[REFERENCE_MODEL][metric]
            - observed[name][metric]
        )

        result[f"{metric}_Difference"] = observed_difference
        result[f"{metric}_CI_Lower"] = lower
        result[f"{metric}_CI_Upper"] = upper

        result[f"{metric}_CI_Excludes_Zero"] = (
            lower > 0 or upper < 0
        )

    rows.append(result)

results = pd.DataFrame(rows)

results.to_csv(OUTPUT_FILE, index=False)

# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

print("\nPAIRED BOOTSTRAP RESULTS")
print("=" * 90)

print(results.round(5).to_string(index=False))

print("\nSaved results to:")
print(OUTPUT_FILE)

print("\nPAIRED BOOTSTRAP COMPLETED SUCCESSFULLY.")
