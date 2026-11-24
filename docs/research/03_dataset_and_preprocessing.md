# 03. Dataset and Preprocessing

## 1. Dataset Overview

This research uses the PhysioNet 2019 Sepsis Challenge dataset for developing and evaluating machine learning models for early sepsis prediction.

The dataset contains hourly ICU observations from individual patients. Each patient record includes physiological measurements, demographic information, hospital and ICU timing information, and a binary sepsis label.

The dataset used in this experiment was obtained from the training data, specifically the `training_setA` directory. The `training_setB` directory contained no patient files in the dataset copy used for this study.

### 1.1 Dataset Statistics

The dataset inspection produced the following statistics:

| Parameter | Value |
|---|---:|
| Total patient files | 11,472 |
| Total hourly records | 445,177 |
| Patients with sepsis | 986 |
| Patients without sepsis | 10,486 |
| Patient-level sepsis prevalence | 8.59% |
| Positive hourly labels | 9,430 |
| Negative hourly labels | 435,747 |
| Positive hourly label percentage | 2.12% |

The dataset exhibits substantial class imbalance, particularly at the hourly observation level.

## 2. Dataset Structure

Each patient is represented by a separate PSV file containing hourly observations.

The inspected sample patient file had 54 rows and 41 columns.

The available variables included:

- HR: Heart Rate
- O2Sat: Oxygen Saturation
- Temp: Body Temperature
- SBP: Systolic Blood Pressure
- MAP: Mean Arterial Pressure
- DBP: Diastolic Blood Pressure
- Resp: Respiratory Rate
- Age: Patient Age
- Gender: Patient Gender
- HospAdmTime: Hospital Admission Time
- ICULOS: Intensive Care Unit Length of Stay
- SepsisLabel: Sepsis Target Label

The dataset also contained laboratory measurements and other clinical variables.

## 3. Class Distribution

The dataset contains substantially more non-sepsis observations than sepsis observations.

At the patient level:

- 986 patients were classified as sepsis patients.
- 10,486 patients were classified as non-sepsis patients.

At the hourly observation level:

- 9,430 observations had a positive sepsis label.
- 435,747 observations had a negative sepsis label.

This imbalance was considered during model development and evaluation.

## 4. Patient-Level Data Splitting

To prevent observations from the same patient appearing in multiple partitions, the dataset was divided at the patient level.

A stratified split was used to maintain a similar proportion of sepsis and non-sepsis patients across the partitions.

| Partition | Patients | Sepsis Patients |
|---|---:|---:|
| Training | 8,030 | 690 |
| Validation | 1,721 | 148 |
| Testing | 1,721 | 148 |
| Total | 11,472 | 986 |

A patient split manifest was maintained to preserve the allocation of patients across the partitions.

The split manifest was used as the reference for the subsequent experimental workflow.

## 5. Missing-Value Analysis

Missing values are an important characteristic of clinical time-series datasets because physiological measurements are not necessarily recorded at every hourly observation.

The initial data inspection identified missing values across several clinical variables.

Temperature was particularly affected, with approximately 66.05% missing values in the inspected dataset.

Rather than removing all observations containing missing values, the preprocessing pipeline retained missingness information through dedicated indicators for selected vital signs.

## 6. Feature Engineering

Feature engineering was performed to represent the current physiological state of patients as well as changes in their measurements over time.

The prepared dataset contained 48 engineered input features.

### 6.1 Original Physiological Features

The following vital signs were included:

- HR
- O2Sat
- Temp
- SBP
- MAP
- Resp

### 6.2 Patient and Hospital Context

The following contextual variables were included:

- Age
- Gender
- ICULOS
- HospAdmTime

### 6.3 Missingness Indicators

Missingness indicators were created for the selected vital signs:

- HR_missing
- O2Sat_missing
- Temp_missing
- SBP_missing
- MAP_missing
- Resp_missing

These indicators allow the models to retain information about the availability of physiological measurements.

### 6.4 Rolling Statistical Features

Rolling means and standard deviations were calculated using 3-hour and 6-hour windows for selected physiological variables.

These features were intended to represent recent physiological levels and variability.

### 6.5 One-Hour Change Features

One-hour changes were calculated for selected vital signs to represent short-term changes in physiological measurements.

### 6.6 Derived Clinical Features

Two additional derived features were included:

- ShockIndex
- ModifiedShockIndex

These features were incorporated into the prepared dataset alongside the original and temporal features.

## 7. Prepared Dataset

After preprocessing and feature engineering, the combined dataset contained:

| Parameter | Value |
|---|---:|
| Total rows | 445,177 |
| Unique patients | 11,472 |
| Total columns | 51 |
| Engineered input features | 48 |
| Additional columns | 3 |

The additional columns were:

- PatientID
- Split
- SepsisLabel

The prepared dataset was saved as:

`prepared_features.csv`

No patient records were skipped during the preparation process.

Missing values were intentionally retained at this stage so that imputation could be performed using training data only.

## 8. Train, Validation, and Test Distribution

The prepared dataset was divided into the following hourly observation partitions:

| Partition | Hourly Records | Positive Hourly Labels |
|---|---:|---:|
| Training | 312,261 | 6,599 |
| Validation | 66,179 | 1,413 |
| Testing | 66,737 | 1,418 |
| Total | 445,177 | 9,430 |

The patient-level split was maintained while generating these partitions.

## 9. Imputation and Scaling

Missing-value handling was performed separately within the model pipelines.

For tabular models, median imputation was fitted using training data and applied to the validation and test partitions.

Scaling was applied where required by the model.

The LSTM pipeline used its own imputation and scaling components. These components were also fitted using training data.

The fitted preprocessing objects were saved alongside the corresponding trained models to support consistent inference.

## 10. LSTM Input Preparation

The Temporal Attention LSTM used a subset of 14 features:

1. HR
2. O2Sat
3. Temp
4. SBP
5. MAP
6. Resp
7. HR_missing
8. O2Sat_missing
9. Temp_missing
10. SBP_missing
11. MAP_missing
12. Resp_missing
13. Age
14. Gender

A sequence length of six hourly observations was used.

The resulting sequence shapes were:

| Partition | Input Shape | Label Shape |
|---|---|---|
| Training | (272111, 6, 14) | (272111,) |
| Validation | (56599, 6, 14) | (56599,) |
| Testing | (59107, 6, 14) | (59107,) |

The sequences were prepared separately for each patient to avoid mixing observations across patient boundaries.

A final missing-value check on the prepared LSTM sequences confirmed that no NaN values remained.

## 11. Summary

The preprocessing pipeline transformed the raw PhysioNet patient files into a structured hourly dataset suitable for machine learning and temporal deep learning experiments.

The pipeline incorporated patient-level data splitting, missingness indicators, rolling statistical features, short-term changes, and derived clinical features.

The final prepared dataset contained 445,177 hourly observations and 48 engineered input features. Separate preprocessing pipelines were used for tabular models and the Temporal Attention LSTM, with preprocessing parameters fitted using training data.

This prepared dataset served as the basis for the subsequent model development and evaluation experiments.