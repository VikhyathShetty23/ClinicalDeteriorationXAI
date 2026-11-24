# ClinicalDeteriorationXAI

**Explainable Temporal Machine Learning for Early Clinical Deterioration Detection**

ClinicalDeteriorationXAI is an interactive clinical decision-support and research prototype platform built with Python and Streamlit. The application explores how machine learning and deep learning models can detect early physiological deterioration in hospitalized and ICU patients using multi-hour longitudinal vital-sign trajectories.

The platform integrates **six research models**, automated temporal feature engineering, missingness-aware imputation, multi-angle explainable AI (SHAP attributions and LSTM temporal self-attention), interactive patient demonstration scenarios, and an empirical research dashboard reflecting held-out benchmark evaluations on the **PhysioNet Computing in Cardiology Challenge 2019** sepsis cohort.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Six-Model Research Architecture](#six-model-research-architecture)
   - [Model Descriptions](#model-descriptions)
   - [Common Test Benchmark Results](#common-test-benchmark-results)
   - [Linear SVM Decision Score Notice](#linear-svm-decision-score-notice)
   - [Feature Engineering & Preprocessing Pipeline](#feature-engineering--preprocessing-pipeline)
3. [Empirical Research Findings](#empirical-research-findings)
   - [Paired Bootstrap Hypothesis Testing](#paired-bootstrap-hypothesis-testing)
   - [Feature Ablation & Timing Confounders](#feature-ablation--timing-confounders)
   - [Early-Warning Lead-Time & Alert Episodes](#early-warning-lead-time--alert-episodes)
4. [Project Structure](#project-structure)
5. [Prerequisites & System Requirements](#prerequisites--system-requirements)
6. [Step-by-Step Installation (Windows)](#step-by-step-installation-windows)
7. [Running the Application](#running-the-application)
8. [Testing & Verification](#testing--verification)
9. [Application Features & Walkthrough](#application-features--walkthrough)
   - [Overview (`app.py`)](#overview-apppy)
   - [New Assessment (`pages/2_New_Assessment.py`)](#new-assessment-pages2_new_assessmentpy)
   - [Prediction Results (`pages/3_Prediction_Results.py`)](#prediction-results-pages3_prediction_resultspy)
   - [Explainability (`pages/4_Explainability.py`)](#explainability-pages4_explainabilitypy)
   - [Trends & Research Dashboard (`pages/5_Trends_Research.py`)](#trends--research-dashboard-pages5_trends_researchpy)
10. [Input Data & CSV Specifications](#input-data--csv-specifications)
11. [Troubleshooting Guide](#troubleshooting-guide)
12. [Important Considerations & Disclaimers](#important-considerations--disclaimers)

---

## Project Overview

Early identification of clinical deterioration (e.g., sepsis onset, hemodynamic collapse, respiratory decompensation) is essential for timely clinical intervention in intensive care units. Traditional rule-based scoring systems (such as NEWS2 or SOFA) offer static snapshots but often struggle with subtle trajectory dynamics and missing observational data.

**ClinicalDeteriorationXAI** evaluates multiple machine learning paradigms against this clinical challenge while preserving transparent interpretability:

- **Six Comparative Architectures**: Original XGBoost, Random Forest, Logistic Regression, Linear SVM, Clinical-Only XGBoost, and Temporal Attention LSTM.
- **Empirical Research Grounding**: All models and benchmarks reflect held-out test evaluations on **58,132 eligible hourly endpoints** across **1,721 patients** with **1,300 positive labels** (~2.24% positive class).
- **Dual Model Suites**: The interface supports the full **Six-Model Research Suite** while maintaining backwards compatibility with the **Three-Model Legacy Suite**.
- **Transparent Explainability**: Individual patient SHAP waterfall/bar attributions, global beeswarm summaries, model split importance, and recurrent LSTM temporal attention distributions across consecutive hourly time steps.
- **Interactive Evaluation**: Manual patient entry, batch CSV upload with schema validation, and pre-built synthetic clinical demonstration scenarios.

---

## Six-Model Research Architecture

### Model Descriptions

| Model Name | Features | Architecture & Details | Output Type | Validation Threshold |
| :--- | :---: | :--- | :---: | :---: |
| **Original XGBoost** | 48 | Gradient-boosted decision trees (`max_depth=4`, `n_estimators=700`, `scale_pos_weight`) | Probability | `0.7345` (artifact) / `0.7487` (benchmark) |
| **Random Forest** | 48 | Ensemble of 300 decision trees (`max_depth=18`, `class_weight="balanced_subsample"`) | Probability | `0.3110` (artifact) / `0.2967` (benchmark) |
| **Logistic Regression** | 48 | L2-regularized linear model with StandardScaler normalization (`class_weight="balanced"`) | Probability | `0.7155` |
| **Linear SVM** | 48 | Linear support vector classifier (`LinearSVC`, $C=0.1$, StandardScaler normalization) | Decision Score | `+0.3732` |
| **Clinical-only XGBoost** | 46 | Gradient-boosted trees explicitly excluding ICU timing variables (`ICULOS`, `HospAdmTime`) | Probability | `0.6528` (artifact) / `0.6630` (benchmark) |
| **Temporal Attention LSTM** | 14 $\times$ 6h | PyTorch LSTM (`hidden_size=32`) with dynamic self-attention over 6 consecutive hourly time steps | Probability | `0.7479` |

### Common Test Benchmark Results

Evaluated on the common eligible test cohort (**58,132 hourly observations**, **1,721 patients**, **1,300 positive deterioration labels**):

| Model Name | AUROC | AUPRC | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Original XGBoost** | **0.7912** | 0.0994 | 0.1413 | 0.3377 | **0.1993** |
| **Linear SVM** | 0.7793 | 0.1015 | 0.1312 | 0.3254 | 0.1870 |
| **Logistic Regression** | 0.7783 | **0.1030** | 0.1334 | 0.3385 | 0.1914 |
| **Random Forest** | 0.7699 | 0.0896 | 0.1327 | **0.3600** | 0.1939 |
| **Clinical-only XGBoost** | 0.7055 | 0.0621 | 0.0856 | 0.2200 | 0.1232 |
| **Temporal Attention LSTM** | 0.6861 | 0.0559 | 0.0789 | 0.1738 | 0.1086 |

> **Note on Thresholds:** The research suite evaluates each model against its saved validation threshold. In the common-test comparison, thresholds were selected on validation data to maximize F1-score.

### Linear SVM Decision Score Notice
`LinearSVC` produces a signed continuous distance from the separating decision boundary via `decision_function()`. **It does not natively produce calibrated probabilities.**
- The application displays the Linear SVM output as a continuous **Decision Score** (e.g., `+0.4215` or `-0.1520`).
- It is compared against its saved decision boundary threshold (`+0.3732`).
- It is **never** presented as a percentage probability or combined with probability meters.

### Feature Engineering & Preprocessing Pipeline
1. **Input Validation**: Enforces at least 6 hourly observations, strictly consecutive ICU hours (`ICULOS` incrementing by 1), and invariant patient demographics (`Age`, `Gender`, `HospAdmTime`).
2. **Missingness Indicator Generation**: Computes binary flags `<vital>_missing = 1` before forward filling, preserving measurement frequency patterns.
3. **Temporal Rolling Statistics**: Computes 3-hour and 6-hour rolling means and standard deviations, 1-hour deltas, Shock Index (`HR / SBP`), and Modified Shock Index (`HR / MAP`).
4. **Model-Specific Selection & Scaling**:
   - Tabular models extract features strictly in the column order saved in their `.joblib` dictionary artifacts.
   - Median imputation (`SimpleImputer`) is applied to all tabular models.
   - `StandardScaler` is applied to Logistic Regression and Linear SVM.
   - LSTM receives sequences of shape `(1, 6, 14)` preprocessed by `six_model_lstm_imputer.joblib` and `six_model_lstm_scaler.joblib`.

---

## Empirical Research Findings

The research experiments in `experiments/six_model_comparison/` established several key clinical findings:

### Paired Bootstrap Hypothesis Testing
Using 1,000 paired non-parametric bootstrap iterations on the common test cohort:
- **Original XGBoost vs Clinical-only XGBoost**: AUROC difference is **+0.0857** (95% CI: `[+0.0503, +0.1195]`). The confidence interval strictly excludes zero, confirming that timing variables significantly elevate discrimination.
- **Original XGBoost vs Temporal Attention LSTM**: AUROC difference is **+0.1051** (95% CI: `[+0.0628, +0.1476]`), statistically significant.
- **Original XGBoost vs Random Forest, Logistic Regression, Linear SVM**: In all three comparisons, the 95% confidence intervals cross zero, indicating no statistically significant difference in discriminative capacity among the 48-feature tabular models.

### Feature Ablation & Timing Confounders
Ablation experiments evaluated the impact of removing operational timing context:
- **Removing `HospAdmTime` alone**: AUROC remains virtually unchanged (**0.7910** vs 0.7912 baseline).
- **Removing `ICULOS` alone**: AUROC drops sharply to **0.7167** (AUPRC drops to 0.0613).
- **Removing Both**: AUROC drops to **0.7055** (AUPRC drops to 0.0621).

*Clinical Takeaway:* Tabular models gain substantial predictive leverage from ICU dwell time (`ICULOS`), as patients remaining in the ICU accumulate higher baseline risk. Clinical-only models isolate physiological signals from this operational shortcut.

### Early-Warning Lead-Time & Alert Episodes
- **Pre-Onset Warning Horizon**: Evaluated at fixed lead times prior to deterioration, Original XGBoost achieved pre-onset detection rates among eligible sepsis patients of **32.74% at 6 hours**, **28.10% at 4 hours**, and **28.91% at 2 hours**.
- **Continuous Warning Window (2–6 Hours)**: 35.16% of eligible patients were detected with a **median earliest warning time of 6.0 hours pre-onset** (mean 5.53 hours).
- **False Alarm Burden**: Non-sepsis controls generated only **5.89 false alert hours per 1,000 patient-hours**.
- **Episode Duration**: Sepsis pre-onset alert episodes averaged **3.54 hours** with **80.8% being sustained** ($\ge 2$ consecutive hours), whereas control patient alerts averaged **1.98 hours** with 55.5% being transient 1-hour spikes.

---

## Project Structure

```text
ClinicalDeteriorationXAI/
│
├── app.py                      # Main entry point & comprehensive research overview
├── requirements.txt            # Python dependencies
├── sample_patient_template.csv # Clean 6-hour CSV template for data entry/upload
├── sample_patient.csv          # Sample 6-hour patient record (clean baseline)
├── sample_missing.csv          # Sample 6-hour patient record with missing measurements
├── missing_patients.csv        # Validation test file (non-consecutive ICU hours)
├── test_six_models.py          # End-to-end integration test for the six-model pipeline
├── test_loader.py              # Test script verifying model loading
├── test_prediction.py          # Test script verifying prediction pipeline
│
├── assets/                     # Custom UI styling
│   └── style.css               # Clinical color palette, cards, typography
│
├── models/                     # Legacy three-model artifacts
│   ├── model_config.joblib
│   ├── xgboost_original.json / .joblib
│   ├── xgboost_clinical_only.json / .joblib
│   ├── improved_lstm.pth
│   └── imputers & scalers
│
├── pages/                      # Multi-page Streamlit interface
│   ├── 1_Overview.py           # Overview placeholder routing to app.py
│   ├── 2_New_Assessment.py     # Assessment page (Manual Entry, CSV Upload, Demo Mode)
│   ├── 3_Prediction_Results.py # Comparative results, Linear SVM card, consensus summary
│   ├── 4_Explainability.py     # Local SHAP, global beeswarm, LSTM attention
│   └── 5_Trends_Research.py    # Dynamic research dashboard (ablation, bootstrap, lead-time)
│
├── utils/                      # Modular backend architecture
│   ├── research_model_loader.py # Loads the six research models & preprocessing bundles
│   ├── research_prediction.py   # Six-model prediction pipeline (probabilities & SVM score)
│   ├── model_loader.py          # Legacy three-model loader
│   ├── prediction.py            # Legacy prediction runner
│   ├── preprocessing.py         # Consecutive hour validation, missingness, rolling stats
│   ├── explainability.py        # TreeExplainer local SHAP & feature importance
│   └── __init__.py
│
└── experiments/
    └── six_model_comparison/   # Research models and reproducible empirical results
        ├── saved_models/       # 9 trained artifacts (XGB, RF, LR, SVM, LSTM, imputers, scalers)
        ├── results/            # common_test_model_comparison.csv, bootstrap, plots, shap/
        ├── ablation/           # ablation_comparison.csv & saved models
        └── lead_time/          # lead_time_summary_full_test.csv, continuous_warning, etc.
```

---

## Prerequisites & System Requirements

- **Operating System**: Windows 10 or 11 (64-bit), macOS, or Linux.
- **Python**: Python 3.10, 3.11, 3.12, or 3.13 (Python 3.13.1 confirmed working).
- **Git**: Git installed for repository cloning.
- **Dependencies**: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `torch`, `shap`, `matplotlib`, `plotly`, `joblib`.

---

## Step-by-Step Installation (Windows)

Open **Windows PowerShell** or **Command Prompt (`cmd`)**:

### 1. Clone the Repository
```bash
git clone https://github.com/VikhyathShetty23/ClinicalDeteriorationXAI.git
cd ClinicalDeteriorationXAI
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
```

### 3. Activate the Virtual Environment
**In PowerShell:**
```powershell
venv\Scripts\Activate.ps1
```
*If PowerShell blocks script execution, run:*
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\Activate.ps1
```

**In Command Prompt (`cmd`):**
```cmd
venv\Scripts\activate.bat
```

### 4. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Application

Ensure your virtual environment is active (`(venv)`), then run:

```bash
streamlit run app.py
```

Streamlit will launch and display the local access URL:
```text
Local URL: http://localhost:8501
```
Open your browser and navigate to `http://localhost:8501`.

To stop the local server, press `Ctrl + C` in your terminal.

---

## Testing & Verification

The repository includes test suites to verify that all models, preprocessing objects, and explainability modules function properly.

Run the complete six-model integration test:
```bash
python test_six_models.py
```
*Expected output: Confirms successful loading of all 6 models, execution of prediction on a 6-hour trajectory, verification of Linear SVM decision score, LSTM attention weights summing to 1.0, and SHAP computation.*

Run the legacy model tests:
```bash
python test_loader.py
python test_prediction.py
```

---

## Application Features & Walkthrough

### Overview (`app.py`)
- Live availability status metrics for all six research models.
- PhysioNet dataset cohort summary (11,472 files, 445,177 hourly records, 986 positive cases).
- Dynamic common-test benchmark table and AUROC progress summaries.
- One-click navigation to run assessments or explore research findings.

### New Assessment (`pages/2_New_Assessment.py`)
- **Model Suite Selector**: Toggle between the complete **Six-Model Research Suite** (default) and the **Three-Model Legacy Suite**.
- **Three Input Modes**:
  1. **Manual Entry**: Input patient age, gender, admission offset, starting ICU hour, and an interactive 6-hour vital sign editor.
  2. **CSV Upload**: Upload a patient observation CSV with preview, schema validation, and template download.
  3. **Demo Mode**: Evaluate realistic pre-built synthetic patient profiles:
     - *Scenario 1: Progressive Deterioration (Tachycardia, Hypotension & Fever)*
     - *Scenario 2: Stable Post-Operative Trajectory (Normotensive, Euthermic)*
     - *Scenario 3: Respiratory Decline with Missing Measurements*
- **Assessment Execution**: Validates data continuity, executes feature engineering, runs inference across all models, and displays summary alert cards.

### Prediction Results (`pages/3_Prediction_Results.py`)
- **Metric Cards**: Predicted probabilities (%) for the 5 probabilistic models.
- **Dedicated Linear SVM Card**: Displays continuous decision score (signed distance), clearly separated from probabilities.
- **Comprehensive Summary Table**: Compares model architectures, score types, predicted values, thresholds, and alert triggers.
- **Side-by-Side Chart**: Bar chart comparing predicted probabilities against model thresholds.
- **Ensemble Consensus**: Summarizes alert agreement (e.g., High Consensus, Split Decision, or Negative Consensus).
- **Submitted Observations**: Expandable table of evaluated hourly records.

### Explainability (`pages/4_Explainability.py`)
- **Local SHAP Explanations**: Select between Original XGBoost (48 features) and Clinical-only XGBoost (46 features) to compute patient-specific TreeExplainer SHAP values.
- **Top 15 Directional Contributions**: Horizontal bar chart color-coded by impact (teal for positive risk increase; terracotta for negative decrease).
- **Global Model Feature Importance**: Built-in split/gain importance for Original XGBoost.
- **LSTM Temporal Attention**: Dynamic bar chart showing the attention weight distribution across the 6 evaluated observation hours.
- **Cohort-Level Research SHAP Visualizations**:
  - Global Mean Absolute SHAP table (`shap_feature_importance.csv`).
  - SHAP summary beeswarm plot (`shap_summary_beeswarm.png`).
  - Global feature importance ranking (`shap_feature_importance.png`).
  - Representative waterfall example (`shap_waterfall_example.png`).

### Trends & Research Dashboard (`pages/5_Trends_Research.py`)
- **Patient Vital-Sign Trends**: Interactive longitudinal plot of patient vitals over ICU observation hours.
- **Six-Model Benchmarks**: Dynamic table and charts loaded directly from `common_test_model_comparison.csv`.
- **Paired Bootstrap Testing**: 95% confidence intervals on AUROC/AUPRC differences from `paired_bootstrap_comparison.csv`.
- **Feature Ablation**: Empirical breakdown of ICULOS vs HospAdmTime removal from `ablation_comparison.csv`.
- **Lead-Time Evaluation**: Pre-onset detection rates at 6h, 4h, and 2h horizons from `lead_time_summary_full_test.csv`.
- **Continuous Warning Analysis**: Earliest warning metrics and false alarm rates from `continuous_warning_summary.csv`.
- **Alert Episode Dynamics**: Sustained vs isolated episode distributions from `alert_episode_summary.csv`.
- **Operating Threshold Sensitivity**: Trade-off curve between sensitivity and alarm fatigue from `threshold_sensitivity_results.csv`.
- **Methodological Limitations**: Detailed discussion of class imbalance, timing confounders, and clinical translation hurdles.

---

## Input Data & CSV Specifications

When uploading a patient file or preparing CSV data, the file must adhere to the following schema:

| Column Name | Type | Description | Allow Missing? |
| :--- | :---: | :--- | :---: |
| `ICULOS` | Integer | ICU Length of Stay (hours). Must be strictly consecutive positive integers without gaps (`diff == 1`). | **No** |
| `Age` | Float / Int | Patient age in years. Must remain constant across all observations. | **No** |
| `Gender` | Integer | Biological sex (`0` = Female, `1` = Male). Constant across observations. | **No** |
| `HospAdmTime`| Float | Hours between hospital admission and ICU admission ($\le 0$). Constant across observations. | **No** |
| `HR` | Float | Heart Rate (beats per minute). | Yes (Imputed) |
| `O2Sat` | Float | Oxygen Saturation (percentage, e.g., 98.0). | Yes (Imputed) |
| `Temp` | Float | Core Temperature (degrees Celsius, e.g., 37.0). | Yes (Imputed) |
| `SBP` | Float | Systolic Blood Pressure (mmHg, e.g., 120.0). | Yes (Imputed) |
| `MAP` | Float | Mean Arterial Pressure (mmHg, e.g., 85.0). | Yes (Imputed) |
| `Resp` | Float | Respiratory Rate (breaths per minute, e.g., 18.0). | Yes (Imputed) |

> **Sample Template:** Download the pre-formatted `sample_patient_template.csv` directly from the **New Assessment** page.

---

## Troubleshooting Guide

### 1. `ModuleNotFoundError: No module named '...'`
- Ensure your virtual environment is active (`(venv)` appears in your prompt).
- Re-run `pip install -r requirements.txt`.

### 2. `FileNotFoundError: Missing required research model artifacts`
- Ensure the `experiments/six_model_comparison/saved_models/` folder contains all 9 required files.
- Run `python test_six_models.py` to diagnose which file is missing.

### 3. PowerShell Script Execution Blocked (`PSSecurityException`)
- Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` in PowerShell, then activate again.
- Alternatively, use Command Prompt: `venv\Scripts\activate.bat`.

### 4. Input Validation Error: `At least 6 hourly observations are required`
- The temporal feature pipeline and recurrent LSTM require at least 6 consecutive hourly records to calculate 3h/6h rolling statistics and populate sequence tensors.

### 5. Input Validation Error: `ICU observations must be consecutive hourly records`
- Verify that `ICULOS` increments by exactly 1 for every row (e.g., 10, 11, 12, 13, 14, 15) without missing or skipped hours.

### 6. Port Collision (`Address already in use`)
- If port `8501` is occupied, launch Streamlit on an alternate port:
  ```bash
  streamlit run app.py --server.port 8502
  ```

---

## Important Considerations & Disclaimers

### Repository Cleanliness
- The virtual environment (`venv/`), Python cache files (`__pycache__/`), and user files are excluded via `.gitignore`.
- Do not commit patient data or machine-specific binaries.

### Data Privacy & Patient Identifiers (HIPAA / PHI)
- **Do not upload or commit real patient health information (PHI).**
- All demonstration scenarios and sample CSVs provided in this repository are synthetic approximations.

### Medical & Research Disclaimer
> [!IMPORTANT]
> **ClinicalDeteriorationXAI is an exploratory research prototype and educational platform.**
> - The models, predictions, and visualizations have **not** been cleared or approved by any medical device regulatory agency (such as the US FDA or CE).
> - Model scores and threshold crossings are decision-support outputs and must **never** be used as a standalone diagnostic device or as the primary basis for patient treatment, escalation, or medication decisions.
> - All outputs must be evaluated in conjunction with professional clinical judgment, patient physical examination, and institutional hospital protocols.
