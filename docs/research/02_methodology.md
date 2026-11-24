# 02. Research Methodology

## 1. Overview

The research follows an experimental machine learning methodology to investigate early sepsis prediction using hourly ICU observations from the PhysioNet 2019 Sepsis Challenge dataset.

The workflow consists of dataset preparation, patient-level data splitting, preprocessing, feature engineering, model development, performance evaluation, explainability, statistical comparison, and early warning analysis.

The experiments are organized to ensure that model comparisons use a common test population and that preprocessing parameters are learned using training data.

## 2. Overall Workflow

The research workflow is divided into the following stages:

1. Dataset collection and inspection.
2. Patient-level train, validation, and test splitting.
3. Missing-value analysis and preprocessing.
4. Physiological and temporal feature engineering.
5. Development of machine learning and temporal deep learning models.
6. Model evaluation using a common test population.
7. SHAP-based explainability.
8. Feature ablation experiments.
9. Patient-level paired bootstrap comparison.
10. Lead-time and continuous warning analysis.
11. Alert episode analysis.
12. Prediction threshold sensitivity analysis.

## 3. Dataset Preparation and Patient-Level Splitting

The PhysioNet 2019 Sepsis Challenge dataset was used for the experiments. It contains hourly physiological observations and corresponding sepsis labels for individual ICU patients.

The dataset was inspected to understand its structure, feature availability, missing values, and class distribution.

A patient-level split was used to prevent observations belonging to the same patient from appearing across different data partitions.

The dataset was divided into:

- Training set: 70%
- Validation set: 15%
- Test set: 15%

The split was stratified at the patient level using sepsis patient status. A patient split manifest was maintained to preserve the allocation of patients across the experimental partitions.

The resulting patient counts were:

| Partition | Patients |
|---|---:|
| Training | 8,030 |
| Validation | 1,721 |
| Testing | 1,721 |
| Total | 11,472 |

The same patient-level split was used for the model comparison experiments.

## 4. Data Preprocessing

Data preprocessing was performed to prepare the hourly observations for machine learning.

The main preprocessing operations included:

- Inspecting missing values in physiological measurements.
- Converting and validating feature values.
- Handling missing values through imputation.
- Scaling numerical features where required.
- Preserving patient identifiers and hourly observation information for evaluation.
- Ensuring that preprocessing parameters were fitted using training data rather than the test data.

Median imputation was used for the tabular machine learning models. The imputer was fitted on training data and subsequently applied to validation and test data.

The LSTM pipeline used its own imputation and scaling components, also fitted using training data.

## 5. Feature Engineering

Feature engineering was performed to represent both the current physiological condition and changes in patient measurements over time.

The engineered features included:

### 5.1 Physiological Features

- Heart Rate (HR)
- Oxygen Saturation (O2Sat)
- Temperature (Temp)
- Systolic Blood Pressure (SBP)
- Mean Arterial Pressure (MAP)
- Respiratory Rate (Resp)

### 5.2 Patient and ICU Context

- Age
- Gender
- ICU Length of Stay (ICULOS)
- Hospital Admission Time (HospAdmTime)

### 5.3 Missingness Indicators

Missingness indicators were generated for selected vital signs to retain information about the availability of measurements.

### 5.4 Temporal Features

Rolling statistical features were calculated over 3-hour and 6-hour windows, including:

- Rolling means
- Rolling standard deviations
- One-hour changes in physiological measurements

### 5.5 Derived Clinical Features

Two derived features were included:

- Shock Index
- Modified Shock Index

The final prepared feature dataset contained 48 engineered input features, in addition to patient identification, split information, and the target label.

## 6. Model Development

Six model configurations were evaluated:

1. Original XGBoost using 48 features.
2. Clinical-only XGBoost using 46 features.
3. Random Forest using 48 features.
4. Logistic Regression using 48 features.
5. Linear Support Vector Machine using 48 features.
6. Temporal Attention LSTM using 14 input features and a sequence length of six hourly observations.

The clinical-only XGBoost configuration excluded ICULOS and HospAdmTime.

The Temporal Attention LSTM was developed to learn patterns across consecutive hourly observations using an LSTM architecture with an attention mechanism.

Each model was trained using the training partition. Validation data was used for model evaluation and threshold selection. The common test population was used for final model comparison.

## 7. Model Evaluation

The models were evaluated using the following metrics:

### 7.1 Area Under the Receiver Operating Characteristic Curve (AUROC)

Measures the model's ability to distinguish between positive and negative classes across classification thresholds.

### 7.2 Area Under the Precision-Recall Curve (AUPRC)

Measures the relationship between precision and recall across thresholds. This metric is particularly relevant for the imbalanced sepsis prediction dataset.

### 7.3 Precision

Measures the proportion of predicted positive observations that were actually positive.

### 7.4 Recall

Measures the proportion of actual positive observations correctly identified by the model.

### 7.5 F1-Score

Represents the harmonic mean of precision and recall.

The final comparison included AUROC, AUPRC, precision, recall, F1-score, and confusion matrix values.

## 8. Explainability Analysis

SHAP (SHapley Additive exPlanations) was used to investigate the contribution of individual features to the Original XGBoost model's predictions.

Mean absolute SHAP values were calculated to estimate global feature importance.

The analysis also included:

- SHAP feature importance ranking.
- SHAP summary beeswarm visualization.
- SHAP waterfall visualization for an individual prediction.

These analyses were used to investigate how physiological, temporal, and contextual features influenced model predictions.

## 9. Feature Ablation Analysis

Feature ablation experiments were conducted to investigate the contribution of ICU and hospital timing information.

The following configurations were compared:

- Original XGBoost.
- XGBoost without ICULOS.
- XGBoost without HospAdmTime.
- XGBoost without both ICULOS and HospAdmTime.

The resulting performance differences were used to assess the importance of these features within the model.

## 10. Statistical Comparison

A paired patient-level bootstrap analysis was conducted to compare the Original XGBoost model against the other evaluated models.

The procedure used 2,000 bootstrap resamples of the test patients.

Patients were sampled with replacement, and the same patient resamples were used for each model comparison. Performance differences in AUROC and AUPRC were calculated, along with 95% confidence intervals.

This analysis was used to investigate whether observed performance differences were distinguishable from zero under the bootstrap procedure.

## 11. Early Warning Analysis

The early warning evaluation focused on the Original XGBoost model using the full test prediction population.

Three complementary analyses were performed:

### 11.1 Exact-Hour Lead-Time Analysis

Predictions were examined at specific lead times of 2, 4, and 6 hours before the first recorded positive sepsis label for each patient.

### 11.2 Continuous Warning-Window Analysis

A warning window covering 2–6 hours before sepsis onset was evaluated.

Patients were considered detected if at least one prediction within this window exceeded the selected threshold.

The analysis also examined the earliest warning time and alert burden among non-sepsis patients.

### 11.3 Alert Episode Analysis

Consecutive positive predictions in consecutive ICU hours were grouped into alert episodes.

Episodes lasting one hour were classified as isolated episodes, while episodes lasting at least two consecutive hours were classified as sustained episodes.

The sepsis analysis was restricted to the 2–6-hour pre-onset window, whereas non-sepsis episodes were evaluated across the full observed stay.

## 12. Threshold Sensitivity Analysis

A threshold sensitivity experiment was conducted using five thresholds:

- 0.65
- 0.70
- 0.748707
- 0.80
- 0.85

For each threshold, the analysis measured:

- Number of sepsis patients detected in the 2–6-hour warning window.
- Detection rate among all sepsis patients.
- Detection rate among eligible sepsis patients.
- Number and percentage of non-sepsis patients receiving alerts.
- False-alert hours per 1,000 observed non-sepsis patient-hours.

The purpose was to investigate the trade-off between early detection and false-alert burden.

The threshold sensitivity analysis was treated as a test-set robustness assessment, not as a method for selecting an optimal threshold.

## 13. Experimental Reproducibility

The experiments were organized into a separate directory to preserve the existing application and model implementation.

Prepared data, model artifacts, prediction files, evaluation metrics, visualizations, and analysis outputs were stored separately.

A fixed patient-level split manifest was maintained, and preprocessing components were fitted using training data to support reproducibility and reduce the risk of information leakage across data partitions.

## 14. Summary

The methodology combines conventional machine learning, temporal deep learning, explainability, feature ablation, statistical comparison, and early warning evaluation.

This experimental design provides a structured approach to investigating predictive performance and model behavior for early sepsis prediction using retrospective ICU data.