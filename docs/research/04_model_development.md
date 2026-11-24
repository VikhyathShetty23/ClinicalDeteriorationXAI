# 04. Model Development

## 1. Introduction

Six model configurations were developed and evaluated for early sepsis prediction using the prepared PhysioNet 2019 dataset.

The experiments included conventional machine learning algorithms and a temporal deep learning architecture. The objective was to compare different learning approaches using a consistent patient-level experimental setup.

The evaluated configurations were:

1. Original XGBoost
2. Clinical-only XGBoost
3. Random Forest
4. Logistic Regression
5. Linear Support Vector Machine
6. Temporal Attention LSTM

## 2. Original XGBoost

XGBoost (Extreme Gradient Boosting) is an ensemble learning algorithm based on gradient-boosted decision trees.

The original XGBoost model used all 48 engineered features, including physiological measurements, demographic information, temporal statistics, missingness indicators, and ICU-related context.

### 2.1 Hyperparameters

| Hyperparameter | Value |
|---|---:|
| Number of estimators | 700 |
| Maximum depth | 4 |
| Learning rate | 0.05 |
| Subsample | 0.8 |
| Column subsample | 0.8 |
| Minimum child weight | 5 |
| L2 regularization (reg_lambda) | 2 |
| Objective | binary:logistic |
| Evaluation metric | aucpr |
| Tree method | hist |
| Random state | 42 |

Class imbalance was addressed using `scale_pos_weight`, calculated from the negative-to-positive class ratio in the training data.

Median imputation was used for missing values. The imputer and model were fitted using training data.

### 2.2 Purpose

The original XGBoost configuration served as the primary tabular model and reference configuration for the feature ablation and explainability experiments.

## 3. Clinical-only XGBoost

A second XGBoost configuration was developed to investigate the predictive contribution of clinical and physiological information without ICU and hospital timing variables.

This configuration used 46 features.

The following features were excluded:

- ICULOS
- HospAdmTime

The remaining features included physiological measurements, demographic variables, missingness indicators, temporal statistics, and derived clinical features.

The model was trained and evaluated using the same patient-level experimental framework.

### Purpose

This configuration helped investigate the contribution of ICU and hospital timing information to predictive performance.

## 4. Random Forest

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees.

The Random Forest model used all 48 engineered features.

The model was included to compare the performance of bagging-based tree ensembles against gradient boosting.

The same training, validation, and test partitions were used.

## 5. Logistic Regression

Logistic Regression is a linear classification algorithm that estimates the probability of a sample belonging to a particular class.

The model used all 48 engineered features.

It provided a comparatively simple baseline for evaluating whether a linear decision function could identify patterns associated with sepsis.

Feature preprocessing, including missing-value handling and scaling, was performed within the model pipeline using training data.

## 6. Linear Support Vector Machine

The Linear Support Vector Machine (Linear SVM) was included to evaluate a margin-based classification approach.

The model used all 48 engineered features.

The linear SVM provided another linear decision boundary baseline for comparison against Logistic Regression and the tree-based models.

The model was trained using the same patient-level training partition and evaluated using the common test population.

## 7. Temporal Attention LSTM

### 7.1 Architecture

A Long Short-Term Memory (LSTM) network with an attention mechanism was developed to learn temporal patterns from consecutive hourly observations.

LSTM networks are designed to process sequential information and capture dependencies across time steps.

The attention mechanism assigns weights to the sequence representations, allowing the model to emphasize different time steps when generating its prediction.

The architecture consisted of:

1. Sequential input layer.
2. LSTM layer.
3. Attention mechanism.
4. Classification layer.

The LSTM used:

| Parameter | Value |
|---|---:|
| Input features | 14 |
| Hidden size | 32 |
| Sequence length | 6 hourly observations |
| Training epochs | 15 |
| Best validation AUPRC epoch | 7 |

### 7.2 Input Features

The LSTM used the following 14 features:

- HR
- O2Sat
- Temp
- SBP
- MAP
- Resp
- HR_missing
- O2Sat_missing
- Temp_missing
- SBP_missing
- MAP_missing
- Resp_missing
- Age
- Gender

The input data was organized into sequences of six consecutive hourly observations for each patient.

### 7.3 Preprocessing

The LSTM pipeline used separate imputation and scaling components.

These preprocessing components were fitted using training data and applied to validation and test data.

The prepared sequences were checked to ensure that no NaN values remained.

### 7.4 Training

The LSTM model was trained for 15 epochs.

The best validation AUPRC was achieved at epoch 7.

The trained model and preprocessing components were saved separately to support reproducible inference.

## 8. Handling Class Imbalance

The dataset contained substantially more negative hourly observations than positive sepsis observations.

Class imbalance was considered during model development.

For the original XGBoost configuration, the `scale_pos_weight` parameter was calculated using the training class distribution.

The models were evaluated using metrics such as AUPRC, precision, recall, and F1-score in addition to AUROC.

This provided a more informative assessment of performance under the imbalanced classification setting.

## 9. Model Threshold Selection

Classification thresholds were selected using validation data according to the model evaluation procedure.

For the original XGBoost model, the training artifact recorded a threshold of 0.734528.

The common six-model comparison used a threshold of 0.748706579208374 for Original XGBoost.

The downstream early warning experiments used the rounded threshold 0.748707 to maintain consistency with the common comparison.

The threshold sensitivity experiment subsequently evaluated additional thresholds as a robustness analysis.

## 10. Saved Model Artifacts

The trained tabular models were saved under the experiment's `saved_models` directory.

The principal model artifacts included:

- `original_xgboost.joblib`
- `clinical_only_xgboost.joblib`
- `random_forest.joblib`
- `logistic_regression.joblib`
- `linear_svm.joblib`

The Temporal Attention LSTM artifacts included:

- `six_model_temporal_attention_lstm.pth`
- `six_model_lstm_imputer.joblib`
- `six_model_lstm_scaler.joblib`
- `six_model_lstm_config.joblib`

The saved preprocessing components were retained alongside the trained models to support consistent prediction.

## 11. Model Comparison Design

All six configurations were evaluated using a common eligible test population of 58,132 hourly endpoints.

This common population contained observations from 1,721 unique test patients and 1,300 positive hourly labels.

Using a common evaluation population allowed model performance metrics to be compared under the same test-label distribution.

The full test partition contained 66,737 hourly observations. The common comparison population was smaller because it retained endpoints eligible for comparison across the six model configurations.

## 12. Summary

The six model configurations represented different approaches to the sepsis prediction problem.

The tabular models provided conventional machine learning baselines, while the Temporal Attention LSTM investigated sequential learning from hourly physiological observations.

The Original XGBoost configuration served as the primary model for subsequent SHAP explainability, feature ablation, and early warning analyses.

The model artifacts and preprocessing components were preserved to support reproducibility and subsequent integration into the Streamlit application.