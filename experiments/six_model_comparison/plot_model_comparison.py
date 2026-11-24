
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(PLOT_DIR, exist_ok=True)

predictions_path = os.path.join(
    RESULTS_DIR, "common_test_predictions.csv"
)

comparison_path = os.path.join(
    RESULTS_DIR, "common_test_model_comparison.csv"
)

predictions = pd.read_csv(predictions_path)
comparison = pd.read_csv(comparison_path)

# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

models = {
    "Original XGBoost": "original_xgboost_score",
    "Clinical Only XGBoost": "clinical_only_xgboost_score",
    "Random Forest": "random_forest_score",
    "Logistic Regression": "logistic_regression_score",
    "Linear SVM": "linear_svm_score",
    "Temporal Attention LSTM": "temporal_attention_lstm_score"
}

# Find the threshold column without depending on capitalization.
threshold_column = next(
    (col for col in comparison.columns
     if col.lower() == "threshold"),
    None
)

if threshold_column is None:
    raise ValueError(
        "Threshold column not found in comparison CSV. "
        f"Available columns: {list(comparison.columns)}"
    )

# Find model-name column.
model_column = next(
    (col for col in comparison.columns
     if col.lower() == "model"),
    None
)

if model_column is None:
    raise ValueError("Model column not found in comparison CSV.")

y_true = predictions["SepsisLabel"].astype(int)

# --------------------------------------------------
# ROC CURVES
# --------------------------------------------------

plt.figure(figsize=(9, 7))

for model_name, score_column in models.items():

    if score_column not in predictions.columns:
        raise ValueError(
            f"Missing prediction column: {score_column}. "
            f"Available columns: {list(predictions.columns)}"
        )

    scores = predictions[score_column]

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{model_name} (AUC = {roc_auc:.3f})"
    )

plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right", fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(PLOT_DIR, "roc_curve.png"),
    dpi=300
)
plt.close()

# --------------------------------------------------
# PRECISION-RECALL CURVES
# --------------------------------------------------

plt.figure(figsize=(9, 7))

for model_name, score_column in models.items():

    scores = predictions[score_column]

    precision, recall, _ = precision_recall_curve(
        y_true, scores
    )

    ap = average_precision_score(y_true, scores)

    plt.plot(
        recall,
        precision,
        linewidth=2,
        label=f"{model_name} (AP = {ap:.3f})"
    )

baseline = y_true.mean()

plt.axhline(
    baseline,
    linestyle="--",
    label=f"Positive prevalence = {baseline:.3f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend(loc="upper right", fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(PLOT_DIR, "precision_recall_curve.png"),
    dpi=300
)
plt.close()

# --------------------------------------------------
# CONFUSION MATRICES
# --------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for ax, (model_name, score_column) in zip(
    axes, models.items()
):

    row = comparison[
        comparison[model_column] == model_name
    ]

    if row.empty:
        raise ValueError(
            f"Threshold not found for {model_name}"
        )

    threshold = float(row.iloc[0][threshold_column])

    y_pred = (
        predictions[score_column] >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No Sepsis", "Sepsis"],
        yticklabels=["No Sepsis", "Sepsis"],
        ax=ax
    )

    ax.set_title(
        f"{model_name}\nThreshold = {threshold:.3f}"
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

plt.suptitle(
    "Confusion Matrix Comparison",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    os.path.join(PLOT_DIR, "confusion_matrices.png"),
    dpi=300
)
plt.close()

# --------------------------------------------------
# METRIC COMPARISON CHART
# --------------------------------------------------

metric_columns = [
    "AUROC",
    "AUPRC",
    "Precision",
    "Recall",
    "F1"
]

available_metrics = [
    col for col in metric_columns
    if col in comparison.columns
]

if available_metrics:

    plot_data = comparison.set_index(
        model_column
    )[available_metrics]

    ax = plot_data.plot(
        kind="bar",
        figsize=(12, 7),
        width=0.8
    )

    ax.set_title("Six Model Performance Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(title="Metrics")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(PLOT_DIR, "metric_comparison.png"),
        dpi=300
    )
    plt.close()

print("\nVISUALIZATION COMPLETED SUCCESSFULLY")
print("------------------------------------")
print("Saved plots in:", PLOT_DIR)

for filename in os.listdir(PLOT_DIR):
    print("-", filename)
