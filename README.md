# ClinicalDeteriorationXAI

**Explainable Temporal Machine Learning for Early Clinical Deterioration Detection**

ClinicalDeteriorationXAI is an interactive clinical decision-support and research prototype application built with Python and Streamlit. The system explores how machine learning (XGBoost) and deep learning (Temporal Attention LSTM) models can detect early physiological deterioration in hospitalized and ICU patients using consecutive hourly vital-sign observations. 

The application incorporates data validation, missing-value imputation, automated temporal feature engineering, multi-model risk assessment, and transparent explainability features—including SHAP (SHapley Additive exPlanations) attribution for gradient-boosted trees and attention weight distributions across consecutive hourly time steps for recurrent neural networks.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Research & Model Architecture](#research--model-architecture)
   - [Original XGBoost Model](#1-original-xgboost-model-48-features)
   - [Clinical-Only XGBoost Model](#2-clinical-only-xgboost-model-46-features)
   - [Temporal Attention LSTM](#3-temporal-attention-lstm-14-features--6-hours)
   - [Data Preprocessing & Feature Pipeline](#data-preprocessing--feature-pipeline)
3. [Project Structure](#project-structure)
4. [Prerequisites and System Requirements](#prerequisites-and-system-requirements)
5. [Step-by-Step Installation (Windows)](#step-by-step-installation-windows)
   - [1. Clone the Repository](#1-clone-the-repository)
   - [2. Create a Virtual Environment](#2-create-a-virtual-environment)
   - [3. Activate the Virtual Environment](#3-activate-the-virtual-environment)
   - [4. Install Required Dependencies](#4-install-required-dependencies)
6. [Trained Model Files](#trained-model-files)
   - [Models Directory Overview](#models-directory-overview)
   - [Verifying Model Files](#verifying-model-files)
   - [Git LFS Guidance](#git-lfs-guidance)
7. [Running the Application](#running-the-application)
   - [Launching the Dashboard](#launching-the-dashboard)
   - [Accessing the Interface](#accessing-the-interface)
   - [Stopping the Application](#stopping-the-application)
   - [Using Sample Datasets](#using-sample-datasets)
8. [Application Features & Walkthrough](#application-features--walkthrough)
   - [Overview (`app.py`)](#overview-apppy)
   - [New Assessment (`pages/2_New_Assessment.py`)](#new-assessment-pages2_new_assessmentpy)
   - [Prediction Results (`pages/3_Prediction_Results.py`)](#prediction-results-pages3_prediction_resultspy)
   - [Explainability (`pages/4_Explainability.py`)](#explainability-pages4_explainabilitypy)
   - [Trends & Research (`pages/5_Trends_Research.py`)](#trends--research-pages5_trends_researchpy)
9. [Troubleshooting](#troubleshooting)
10. [Important Considerations & Disclaimers](#important-considerations--disclaimers)
11. [License & Acknowledgments](#license--acknowledgments)

---

## Project Overview

Early detection of clinical deterioration (such as sepsis onset, respiratory decompensation, and hemodynamic instability) is vital for timely intervention in intensive care units. Traditional scoring protocols (e.g., NEWS2, SOFA) provide intermittent snapshots but often struggle to capture complex non-linear interactions and subtle trajectory trends.

**ClinicalDeteriorationXAI** investigates how supervised machine learning models can assist clinical teams by analyzing temporal vitals while keeping model behavior interpretable.

### Implemented Highlights
- **Comparative Multi-Model Pipeline**: Simultaneously evaluates patient state using two gradient-boosted decision tree configurations (Original XGBoost and Clinical-Only XGBoost) and a deep recurrent model (Temporal Attention LSTM).
- **Temporal & Physiological Feature Engineering**: Computes rolling window statistics (3-hour and 6-hour means and standard deviations), 1-hour rate-of-change deltas, Shock Index (`HR / SBP`), and Modified Shock Index (`HR / MAP`).
- **Missingness-Aware Imputation**: Recognizes clinical observation patterns by extracting missingness indicators (`<vital>_missing`) prior to forward filling and median/mean imputation.
- **Explainable AI (XAI)**:
  - **Local Feature Contributions**: Individual patient SHAP values via `shap.TreeExplainer` visualized as directional waterfall/horizontal bar charts.
  - **Global Feature Importance**: XGBoost model-level feature importance metrics.
  - **Temporal Attention Attribution**: Visual distribution of LSTM attention weights across 6 consecutive hourly time steps.
- **Interactive Multi-Page Streamlit Dashboard**: Provides streamlined interfaces for manual entry, batch CSV upload, comparative risk threshold evaluation, and longitudinal vital-sign trend analysis.

---

## Research & Model Architecture

The experimental models in this project were trained on an experimental subset of the **PhysioNet Computing in Cardiology Challenge 2019** sepsis prediction dataset (11,472 patient files, 445,177 hourly observations, 986 positive label cases, with an overall hourly positive class prevalence of ~2.12%).

```
Patient Observation Sequence (≥ 6 Hours)
   │
   ├──> Input Validation (Consecutive ICULOS, Range & Demographics Checks)
   │
   ├──> Missingness Indicators & Forward Fill (ffill)
   │
   ├──> Temporal Feature Engineering (Rolling 3h/6h Mean & Std, Diff, Shock Indices)
   │
   ├───┬───────────────────────┬───────────────────────────────┐
   │   │                       │                               │
   ▼   ▼                       ▼                               ▼
[Original XGBoost]     [Clinical-Only XGBoost]     [Temporal Attention LSTM]
 48 Features            46 Features                 14 Features × 6 Timesteps
 (Includes ICULOS &     (Excludes ICULOS &          (Sequence of 6 consecutive
  HospAdmTime)           HospAdmTime)                hours with Attention)
   │                       │                               │
   ▼                       ▼                               ▼
 Prob: AUROC 0.7767      Prob: AUROC 0.6759              Prob: AUROC 0.6968
 Threshold: 0.6990       Threshold: 0.6361               Threshold: 0.7521
   │                       │                               │
   └───────────────────────┼───────────────────────────────┘
                           │
                           ▼
          Interactive Comparative Dashboard & XAI
     (SHAP Local Impact, Global Importance, Attention Weights)
```

### 1. Original XGBoost Model (48 Features)
- **Role**: High-capacity gradient-boosted decision tree baseline evaluating the patient's state at the latest hourly observation.
- **Feature Set (48 total)**:
  - 6 raw vital signs: Heart Rate (`HR`), Oxygen Saturation (`O2Sat`), Temperature (`Temp`), Systolic Blood Pressure (`SBP`), Mean Arterial Pressure (`MAP`), Respiration Rate (`Resp`).
  - 4 context variables: `Age`, `Gender` (0 = Female, 1 = Male), `ICULOS` (ICU Length of Stay in hours), `HospAdmTime` (hours elapsed between hospital and ICU admission).
  - 6 missingness indicators: Binary flags indicating whether each vital was unmeasured at that hour.
  - 30 rolling temporal features: 3-hour mean, 3-hour standard deviation, 6-hour mean, 6-hour standard deviation, and 1-hour delta for each of the 6 vitals.
  - 2 physiological ratios: Shock Index (`HR / SBP`) and Modified Shock Index (`HR / MAP`).
- **Research Evaluation Metrics**: AUROC **0.7767** | AUPRC **0.1016** | Precision **0.1201** | Recall **0.3748** | F1-Score **0.1819**.
- **Decision Threshold**: `0.6990`.

### 2. Clinical-Only XGBoost Model (46 Features)
- **Role**: Designed to evaluate whether administrative/operational timing variables artificially inflate model performance.
- **Feature Set (46 total)**: Identical to the Original XGBoost feature set, but explicitly removes `ICULOS` and `HospAdmTime`. Retains all physiological vitals, demographics (`Age`, `Gender`), missingness indicators, rolling statistics, and shock indices.
- **Research Evaluation Metrics**: AUROC **0.6759** | AUPRC **0.0573** | Precision **0.0750** | Recall **0.2427** | F1-Score **0.1146**.
- **Decision Threshold**: `0.6361`.
- **Finding**: Removing timing variables results in an AUROC drop from 0.7767 to 0.6759, revealing that ICU length of stay is an influential confounding factor in raw tabular deterioration models.

### 3. Temporal Attention LSTM (14 Features $\times$ 6 Hours)
- **Role**: Deep recurrent neural network modeling physiological trends directly across a sequence of 6 consecutive hourly observations.
- **Architecture**:
  - Input layer: Sequences of shape `(batch_size, 6, 14)` consisting of the 6 vitals, 6 missingness flags, `Age`, and `Gender`.
  - Recurrent layer: PyTorch `nn.LSTM(input_size=14, hidden_size=32, batch_first=True)`.
  - Attention layer: Linear self-attention layer `nn.Linear(32, 1)` followed by Softmax along the sequence dimension, producing attention weights that sum to 1.0 across the 6 time steps.
  - Context aggregation: Attention-weighted sum of LSTM hidden states.
  - Classifier head: `Linear(32, 16) -> ReLU -> Dropout(0.3) -> Linear(16, 1) -> Sigmoid`.
- **Research Evaluation Metrics**: AUROC **0.6968** | AUPRC **0.0563** | Precision **0.0967** | Recall **0.1553** | F1-Score **0.1192**.
- **Decision Threshold**: `0.7521`.

### Data Preprocessing & Feature Pipeline
Implemented in `utils/preprocessing.py`:
1. **Input Validation**: Ensures at least 6 hourly observations are provided, `ICULOS` values are consecutive positive integers without gaps (`diff == 1`), and static demographic variables (`Age`, `Gender`, `HospAdmTime`) remain consistent.
2. **Missingness Indicator Generation**: Computes `<vital>_missing = 1` if absent, preserving the clinical reality that the frequency of lab/vital measurements is itself an informative predictor.
3. **Forward Fill & Imputation**: Applies forward filling (`ffill`) on past measurements within the sequence. Any residual missing values at the beginning of the sequence are imputed using fitted scikit-learn `SimpleImputer` instances.
4. **Feature Scaling**: Applies fitted scikit-learn `StandardScaler` transformations to continuous features before passing sequence tensors to the PyTorch LSTM.

---

## Project Structure

```
ClinicalDeteriorationXAI/
├── app.py                      # Main application entry point & Overview dashboard
├── requirements.txt            # Python package dependencies
├── .gitignore                  # Git ignore rules (venv, cache, IDE files)
├── sample_patient.csv          # Sample 6-hour patient record (clean baseline)
├── sample_missing.csv          # Sample 6-hour patient record with missing measurements
├── missing_patients.csv        # Validation test file (non-consecutive ICU hours)
├── check_model.py              # Script inspecting PyTorch LSTM checkpoint layers
├── check_model_files.py        # Verification utility checking all model & imputer files
├── check_shap.py               # Verification utility testing SHAP TreeExplainer execution
├── check_xgb.py                # Verification utility checking XGBoost model files
├── test_loader.py              # Test script checking model_loader.py execution
├── test_prediction.py          # End-to-end test verifying multi-model prediction pipeline
│
├── assets/                     # Front-end styling assets
│   └── style.css               # Custom CSS for Streamlit (fonts, cards, color palette)
│
├── models/                     # Trained weights, imputers, scalers & configurations
│   ├── model_config.joblib     # Dictionary of feature lists, sequence length, and thresholds
│   ├── xgboost_original.json   # Original XGBoost model (portable JSON format)
│   ├── xgboost_original.joblib # Original XGBoost model (Joblib serialization)
│   ├── xgboost_clinical_only.json   # Clinical-only XGBoost model (portable JSON format)
│   ├── xgboost_clinical_only.joblib # Clinical-only XGBoost model (Joblib serialization)
│   ├── improved_lstm.pth       # PyTorch state dictionary for TemporalAttentionLSTM
│   ├── xgboost_imputer.joblib  # SimpleImputer for the 48-feature XGBoost pipeline
│   ├── clinical_imputer.joblib # SimpleImputer for the 46-feature clinical pipeline
│   ├── lstm_imputer.joblib     # SimpleImputer for the 14-feature LSTM pipeline
│   └── lstm_scaler.joblib      # StandardScaler for LSTM sequence features
│
├── pages/                      # Streamlit multi-page interface views
│   ├── 1_Overview.py           # Overview placeholder (routes to app.py)
│   ├── 2_New_Assessment.py     # Patient assessment page (manual entry & CSV upload)
│   ├── 3_Prediction_Results.py # Comparative model prediction and threshold alert page
│   ├── 4_Explainability.py     # XAI page (SHAP feature attribution & LSTM attention)
│   └── 5_Trends_Research.py    # Longitudinal vital-sign trends & research evaluation
│
└── utils/                      # Core application backend modules
    ├── __init__.py             # Package marker
    ├── model_loader.py         # Model loading, PyTorch LSTM architecture, and device setup
    ├── preprocessing.py        # Data validation, missingness flags, rolling stats & pipelines
    ├── prediction.py           # Multi-model inference, probability scoring & alert generation
    └── explainability.py       # SHAP TreeExplainer integration & XGBoost feature importance
```

### Purpose of Key Files & Folders
- **`app.py`**: The primary Streamlit application script. It initializes application settings, injects custom CSS styling, loads trained models into `st.session_state`, and presents project metrics, dataset statistics, and navigation links.
- **`models/`**: Houses all serialized model artifacts, preprocessing imputers, standard scalers, and configuration dictionaries required for inference without requiring retraining.
- **`pages/`**: Streamlit's native routing directory. Each script represents a dedicated page in the web interface.
- **`utils/`**: Modularized Python logic separating data validation (`preprocessing.py`), checkpoint loading (`model_loader.py`), prediction logic (`prediction.py`), and explainability computations (`explainability.py`).
- **`assets/style.css`**: Provides custom typography (`Outfit` and `DM Sans`), card containers, styled sidebar navigation, and clinical color themes (`#172b4d` navy, `#367f7a` teal).
- **`sample_*.csv`**: Ready-to-use CSV files to test standard execution, missing-value imputation, and validation constraints without needing manual data entry.

---

## Prerequisites and System Requirements

### Hardware & Software Requirements
- **Operating System**: Windows 10 or Windows 11 (64-bit recommended). Also compatible with macOS and Linux.
- **Python**: **Python 3.10, 3.11, 3.12, or 3.13** (64-bit).
  > **Note**: Python 3.13.1 has been confirmed working with the dependencies in `requirements.txt`. Ensure `python --version` outputs a compatible 64-bit Python installation.
- **Git**: Git for Windows installed and accessible from your terminal (`git --version`).
- **Development Tool (Optional)**: [Visual Studio Code](https://code.visualstudio.com/) with the Python extension.

### Key Python Libraries (`requirements.txt`)
- `streamlit`: Multi-page web framework for interactive dashboards.
- `pandas` & `numpy`: High-performance tabular data structures and numerical operations.
- `scikit-learn`: Data preprocessing transformers (`SimpleImputer`, `StandardScaler`).
- `xgboost`: Gradient-boosted decision tree framework for tabular deterioration modeling.
- `torch`: PyTorch deep learning framework for the recurrent LSTM with temporal attention.
- `shap`: SHapley Additive exPlanations for tree-based feature attribution.
- `matplotlib` & `plotly`: Scientific plotting libraries for SHAP waterfalls, attention distributions, and vital trends.
- `joblib`: Serialization and deserialization of scikit-learn transformers and configuration files.

---

## Step-by-Step Installation (Windows)

Follow these step-by-step instructions in **Windows PowerShell** or **Command Prompt (`cmd`)**.

### 1. Clone the Repository
Open PowerShell or Command Prompt, navigate to your desired working directory, and clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd ClinicalDeteriorationXAI
```

> **Note**: Replace `YOUR_REPOSITORY_URL` with the actual URL of your Git repository (e.g., `https://github.com/your-username/ClinicalDeteriorationXAI.git`).

### 2. Create a Virtual Environment
Isolate the project dependencies from your global Python environment by creating a dedicated virtual environment named `venv`:

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

#### On Windows PowerShell:
```powershell
venv\Scripts\Activate.ps1
```

#### If PowerShell Blocks Script Execution:
Windows PowerShell restricts script execution by default (`PSSecurityException`). If you encounter an error stating that *running scripts is disabled on this system*, update your execution policy for the current user:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

After running this command, activate the environment again:

```powershell
venv\Scripts\Activate.ps1
```

#### On Windows Command Prompt (`cmd.exe`):
```cmd
venv\Scripts\activate.bat
```

When successfully activated, your terminal prompt will display the environment prefix: `(venv)`.

### 4. Install Required Dependencies
Upgrade `pip` to the latest version and install all packages listed in `requirements.txt`:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify that all dependencies installed correctly by listing them:

```bash
pip list
```

---

## Trained Model Files

### Models Directory Overview
All trained model checkpoints, imputation transformers, and configuration mappings are stored directly inside the `models/` directory:

| Filename | Format | Size | Purpose |
| :--- | :--- | :--- | :--- |
| `model_config.joblib` | Joblib | ~1 KB | Feature names (48 XGB, 46 Clinical, 14 LSTM), sequence length (`6`), hidden units (`32`), and calibrated decision thresholds |
| `xgboost_original.json` | JSON | ~1.1 MB | Full XGBoost model weights (portable native XGBoost format) |
| `xgboost_original.joblib` | Joblib | ~793 KB | Full XGBoost model (Joblib serialization) |
| `xgboost_clinical_only.json` | JSON | ~1.1 MB | Clinical-only XGBoost model weights (portable native XGBoost format) |
| `xgboost_clinical_only.joblib` | Joblib | ~796 KB | Clinical-only XGBoost model (Joblib serialization) |
| `improved_lstm.pth` | PyTorch | ~31 KB | Trained PyTorch state dictionary for `TemporalAttentionLSTM` |
| `xgboost_imputer.joblib` | Joblib | ~1.8 KB | Fitted `SimpleImputer` for the 48-feature XGBoost pipeline |
| `clinical_imputer.joblib` | Joblib | ~1.7 KB | Fitted `SimpleImputer` for the 46-feature clinical pipeline |
| `lstm_imputer.joblib` | Joblib | ~1 KB | Fitted `SimpleImputer` for the 14-feature LSTM pipeline |
| `lstm_scaler.joblib` | Joblib | ~1 KB | Fitted `StandardScaler` for LSTM input features |

### Verifying Model Files
The repository includes dedicated verification scripts to ensure that all model weights and transformers load correctly before launching the web server.

Run the file verification check:
```bash
python check_model_files.py
```
*Expected output: `OK` for all joblib files and `improved_lstm.pth`.*

Run the model loader check:
```bash
python test_loader.py
```
*Expected output: Confirms successful instantiation of both XGBoost classifiers, the PyTorch LSTM model, and CPU/CUDA device assignment.*

Run the end-to-end inference test:
```bash
python test_prediction.py
```
*Expected output: Computes probabilities and alert triggers on a synthetic 6-hour patient trajectory across all three models.*

### Git LFS Guidance
The total size of the `models/` directory is **under 5 MB**, and each individual file is smaller than 1.2 MB. Because GitHub allows individual files up to 100 MB, these models are tracked directly within standard Git without requiring external downloads.

However, if your fork or organization utilizes **Git Large File Storage (Git LFS)** to manage binary files, ensure Git LFS is initialized before cloning:

```bash
git lfs install
git lfs pull
```

> **Tip**: If `python check_model_files.py` fails with an `UnpicklingError` or EOF error, your files may be Git LFS text pointer files rather than binary weights. Running `git lfs pull` resolves this by downloading the actual binary files.

---

## Running the Application

### Launching the Dashboard
Ensure your virtual environment is active (`(venv)`), then run the application entry point using Streamlit:

```bash
streamlit run app.py
```

### Accessing the Interface
Once launched, Streamlit will display local and network URLs in your terminal:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open your browser and navigate to `http://localhost:8501`.

### Stopping the Application
To stop the Streamlit local server, return to your terminal and press:
```bash
Ctrl + C
```

### Using Sample Datasets
The repository provides three sample CSV files in the project root to facilitate immediate testing:
1. **`sample_patient.csv`**: Contains 6 consecutive ICU hourly observations (ICULOS 10 to 15) for a 65-year-old male with fluctuating vitals. Use this on the **New Assessment** page via the **CSV Upload** tab.
2. **`sample_missing.csv`**: Contains 6 consecutive ICU observations with missing values in `Temp` and `O2Sat`. Upload this file to observe the missingness indicator extraction and forward filling pipeline in action.
3. **`missing_patients.csv`**: Contains non-consecutive ICU hours (skips hour 12). Uploading this file tests the application's built-in input validation constraints.

---

## Application Features & Walkthrough

The web application consists of five integrated views accessible via the sidebar navigation:

### Overview (`app.py`)
- **System Status**: Displays live metrics confirming that the Original XGBoost, Clinical-only XGBoost, and Temporal Attention LSTM models are loaded into session memory.
- **Dataset Summary**: Summarizes research cohort figures from the PhysioNet 2019 dataset subset (11,472 patient files, 445,177 hourly records, 986 positive label cases).
- **Benchmark Performance**: Summarizes held-out test AUROC scores for all three models (Original XGBoost: 0.7767, Clinical-only XGBoost: 0.6759, Improved LSTM: 0.6968).
- **Direct Navigation**: Provides one-click access to initiate a new patient assessment.

### New Assessment (`pages/2_New_Assessment.py`)
Supports two flexible input workflows:
- **Manual Entry Tab**:
  - Demographics input: Patient age (18–120), gender selection (Female = 0, Male = 1), and hospital admission time offset (`HospAdmTime`).
  - Observation table editor: Interactive tabular grid (`st.data_editor`) pre-populated with 6 consecutive ICU hours starting from a customizable hour (`ICULOS`).
  - Measurement entry: Input Heart Rate (`HR`), Oxygen Saturation (`O2Sat`), Temperature (`Temp`), Systolic Blood Pressure (`SBP`), Mean Arterial Pressure (`MAP`), and Respiration Rate (`Resp`). Unavailable measurements can be left blank.
- **CSV Upload Tab**:
  - Upload a single patient's longitudinal CSV file containing the required column headers (`HR`, `O2Sat`, `Temp`, `SBP`, `MAP`, `Resp`, `Age`, `Gender`, `ICULOS`, `HospAdmTime`).
  - Automatically displays a preview of the submitted observations.
- **Assessment Execution**: Validates data continuity, executes feature engineering, runs inference across all three models, and saves results into `st.session_state`.

### Prediction Results (`pages/3_Prediction_Results.py`)
- **Model Comparison Cards**: Displays predicted deterioration probabilities (%) alongside visual progress meters and thresholds for each model.
- **Threshold Alert Status**: Flags whether each model's prediction crossed its calibrated alert threshold:
  - Original XGBoost threshold: `69.90%`
  - Clinical-only XGBoost threshold: `63.61%`
  - Temporal Attention LSTM threshold: `75.21%`
- **Model Agreement Interpretation**: Analyzes whether all models agree, split decisions occur, or all models remain below alert thresholds.
- **Input Data Summary**: Displays an expandable view of the submitted observation table and ICU observation span.

### Explainability (`pages/4_Explainability.py`)
- **Individual SHAP Contributions**:
  - Select between the Original XGBoost model (48 features) and the Clinical-only XGBoost model (46 features).
  - Uses `shap.TreeExplainer` to compute exact local SHAP attributions for the current patient assessment.
  - Renders a horizontal bar chart of the **Top 15 Individual SHAP Contributions** with color-coded directional impact (teal for positive contributions that increase predicted deterioration risk; terracotta for negative contributions that reduce risk).
  - Expandable table presenting the full list of all SHAP values and the model's base expected value.
- **Built-in Model Feature Importance**:
  - Renders the top 15 features based on the trained XGBoost model's split/gain importance, contrasting global model-level weighting against local case-level explanations.
- **LSTM Temporal Attention Distribution**:
  - Visualizes the attention weights assigned by the LSTM across the 6 consecutive ICU observation hours.
  - Identifies which specific hourly observation contributed most heavily to the recurrent neural network's final logit.

### Trends & Research (`pages/5_Trends_Research.py`)
- **Interactive Vital-Sign Trajectories**: Select any vital sign (`HR`, `O2Sat`, `Temp`, `SBP`, `MAP`, `Resp`) to inspect its longitudinal trend plotted across consecutive ICU observation hours.
- **Detailed Research Benchmark Metrics**: Complete performance comparison on the held-out test cohort:
  
  | Model | AUROC | AUPRC | Precision | Recall | F1-Score |
  | :--- | :---: | :---: | :---: | :---: | :---: |
  | **Original XGBoost** | 0.7767 | 0.1016 | 0.1201 | 0.3748 | 0.1819 |
  | **Clinical-only XGBoost** | 0.6759 | 0.0573 | 0.0750 | 0.2427 | 0.1146 |
  | **Improved LSTM** | 0.6968 | 0.0563 | 0.0967 | 0.1553 | 0.1192 |

- **Exploratory Early-Warning Analysis**:
  - Evaluates alert timing relative to estimated sepsis onset within an exploratory window (12 hours before to 3 hours after estimated onset).
  - Highlights that the Original XGBoost model generated pre-onset alerts for **62 patients** with a **median useful lead time of 12 hours** (6 alerts at/after onset, 5 too early, 75 no alert).
- **Research Limitations**: Clear documentation on class imbalance (~2.12% positive rate), ICU timing dependencies, exploratory NEWS2 comparisons, and lack of prospective clinical validation.

---

## Troubleshooting

### 1. `ModuleNotFoundError: No module named '...'`
- **Cause**: The virtual environment is either not activated or dependencies were installed into a different Python environment.
- **Solution**: Check that your prompt indicates `(venv)`. Re-activate the environment:
  ```powershell
  venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```

### 2. `FileNotFoundError: Missing model files: [...]`
- **Cause**: One or more required files in the `models/` directory are missing or misnamed.
- **Solution**: Verify that all 10 files listed in the [Trained Model Files](#trained-model-files) table are present in `models/`. Run `python check_model_files.py` to identify which file is missing.

### 3. PowerShell Script Execution Error (`PSSecurityException`)
- **Cause**: Windows PowerShell security policy blocks the execution of unsigned local scripts.
- **Solution**: Execute the following command in PowerShell and then reactivate:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  venv\Scripts\Activate.ps1
  ```
  Alternatively, activate using standard Command Prompt (`cmd.exe`): `venv\Scripts\activate.bat`.

### 4. Input Validation Error: `At least 6 hourly observations are required` or `Observations must cover consecutive ICU hours`
- **Cause**: The application requires a sequence of at least 6 consecutive hourly records to calculate 6-hour rolling statistics and populate the LSTM sequence input.
- **Solution**: Ensure your input contains at least 6 rows and that the `ICULOS` values increment by exactly 1 for each row (e.g., 10, 11, 12, 13, 14, 15) without skipped hours.

### 5. Streamlit Port Conflict or Address Already in Use
- **Cause**: Another service or an earlier Streamlit session is already occupying default port `8501`.
- **Solution**: Launch Streamlit on an alternative port:
  ```bash
  streamlit run app.py --server.port 8502
  ```

### 6. Git LFS Pointer File / Deserialization Failure
- **Cause**: Model files were cloned as Git LFS pointer text files rather than actual binary artifacts.
- **Solution**: Install Git LFS and pull the binary payloads:
  ```bash
  git lfs install
  git lfs pull
  ```

---

## Important Considerations & Disclaimers

### 1. Repository Cleanliness
- **Virtual Environments**: Do not commit the `venv/` directory to Git. Virtual environments contain machine-specific binaries and paths. The repository's `.gitignore` is already pre-configured to exclude `venv/`, `.venv/`, and Python bytecode caches (`__pycache__/`).
- **Dependencies**: Always manage packages using `requirements.txt`. If you install new libraries, update the requirements file using `pip freeze > requirements.txt` (ensuring local paths are omitted).

### 2. Data Privacy & Patient Identifiers (HIPAA / PHI)
- Never commit private clinical data, Protected Health Information (PHI), or un-anonymized medical records to this repository.
- The sample files provided (`sample_patient.csv`, `sample_missing.csv`) contain synthetic records created purely for software testing.

### 3. Medical & Research Disclaimer
> [!IMPORTANT]
> **ClinicalDeteriorationXAI is an exploratory research prototype and educational decision-support tool.**
> - The predictive models and explainability visualizations have **not** been cleared or approved by any medical device regulatory agency (such as the FDA or CE).
> - Predictions and risk probabilities generated by this software must **never** be used as a standalone diagnostic tool or as the primary basis for patient management, escalation, or clinical treatment decisions.
> - All outputs must be evaluated in conjunction with comprehensive clinical assessments, standard medical protocols, and professional clinical judgment.

---

## License & Acknowledgments

- **PhysioNet Challenge 2019**: This project utilizes an experimental subset derived from the *PhysioNet Computing in Cardiology Challenge 2019: Early Prediction of Sepsis from Clinical Data* (Reyna et al.).
- **Libraries**: Built with open-source tools including [Streamlit](https://streamlit.io/), [PyTorch](https://pytorch.org/), [XGBoost](https://xgboost.readthedocs.io/), and [SHAP](https://shap.readthedocs.io/).
