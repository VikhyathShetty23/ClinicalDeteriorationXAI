# 09. Limitations and Future Work

## 1. Introduction

Although the experiments provide insights into machine learning approaches for early sepsis prediction, several limitations must be considered when interpreting the results.

The study uses retrospective ICU data and evaluates model performance under a specific experimental setup. The findings should therefore be interpreted as experimental results rather than evidence of clinical effectiveness.

This section discusses the limitations of the dataset, preprocessing pipeline, model development, explainability analysis, statistical evaluation, and early warning experiments. It also identifies opportunities for future research.

## 2. Dataset Limitations

### 2.1 Class Imbalance

The dataset contains substantially more negative observations than positive sepsis observations.

At the hourly observation level, only 2.12% of the records were positive.

This imbalance makes model evaluation challenging because a model may achieve reasonable overall discrimination while still having limited positive-class detection performance.

Although multiple evaluation metrics were used, the relatively low precision and F1-scores indicate that identifying positive observations remains challenging.

### 2.2 Missing Clinical Measurements

Several physiological measurements contain missing values.

Temperature was particularly affected, with approximately 66.05% missing values in the inspected dataset.

Although missingness indicators and imputation were incorporated into the preprocessing pipeline, imputation may not fully represent the underlying reasons for missing clinical measurements.

Future work could investigate more advanced missing-data strategies and evaluate their effects on model performance.

### 2.3 Dataset Generalizability

The experiments were conducted using the PhysioNet 2019 Sepsis Challenge dataset, specifically the training data available in the dataset copy used for this research.

The results may not generalize directly to other hospitals, healthcare systems, or patient populations.

External validation using independent clinical datasets is required before making claims about broader applicability.

## 3. Model Development Limitations

### 3.1 Model Performance

The six-model comparison showed that Original XGBoost achieved the highest AUROC and F1-score, while Logistic Regression achieved the highest AUPRC.

However, the overall precision and F1-scores were relatively low across the evaluated models.

This indicates that the models still face challenges in accurately identifying positive hourly sepsis observations while controlling false-positive predictions.

### 3.2 Temporal Attention LSTM Performance

The Temporal Attention LSTM underperformed the tabular models in the current experimental setup.

This does not establish that temporal deep learning is inherently unsuitable for sepsis prediction.

The result may depend on the selected input features, sequence length, architecture, training configuration, class imbalance, and available training data.

Future experiments could investigate alternative sequence lengths, architectures, temporal feature representations, and training strategies.

### 3.3 Feature Selection and Temporal Information

The feature ablation experiment demonstrated that removing ICULOS substantially reduced Original XGBoost performance.

SHAP analysis also identified ICULOS as the most influential feature.

Although this indicates that ICU timing information contributes significantly to the model predictions, the exact reason for this contribution requires further investigation.

The model may be using temporal information associated with the progression of patient observations rather than only physiological indicators.

## 4. Explainability Limitations

SHAP analysis was performed using 1,000 test observations from the Original XGBoost model.

Although this provided useful global feature importance and individual prediction explanations, the findings have limitations.

1. The SHAP analysis was performed on a sample rather than the entire test population.
2. SHAP values describe contributions to model predictions and do not establish causal relationships.
3. High feature importance does not necessarily indicate clinical relevance.
4. Explanations from Original XGBoost cannot automatically be generalized to other model architectures.
5. The strong contribution of ICU timing variables requires additional investigation.

Future work could investigate explanation stability across different patient samples and compare explanation methods across multiple model architectures.

## 5. Statistical Analysis Limitations

The paired patient-level bootstrap analysis used 2,000 resamples and calculated 95% confidence intervals for AUROC and AUPRC differences.

However, the analysis has several limitations:

- Confidence intervals were not adjusted for multiple comparisons.
- Comparisons were restricted to the evaluated models and common test population.
- Bootstrap confidence intervals do not establish generalizability to independent external populations.
- Confidence intervals containing zero do not prove that two models are equivalent.

Future research could investigate additional statistical comparison methods and independent validation datasets.

## 6. Early Warning Analysis Limitations

### 6.1 Recorded Sepsis Onset

Sepsis onset was defined using the first recorded positive SepsisLabel observation.

This represents the recorded dataset label transition and may not correspond exactly to the clinical onset of sepsis.

Therefore, the calculated lead times should be interpreted relative to the recorded onset rather than a confirmed clinical onset time.

### 6.2 Missing Warning-Window Observations

Some sepsis patients did not have observations available at the specified lead times or within the continuous 2–6-hour warning window.

In the continuous warning analysis, 20 of the 148 sepsis patients had no observations available in the window.

This limits the number of patients eligible for early warning evaluation.

### 6.3 Warning-Window Boundary

A substantial proportion of detected patients had their earliest warning at the six-hour boundary.

This means that the actual beginning of warning activity may have occurred earlier than the evaluated period.

The analysis cannot establish the full duration of these warnings beyond the six-hour boundary.

### 6.4 Alert Episode Comparison

The alert episode analysis used different observation periods for the two patient groups.

Sepsis episodes were restricted to the 2–6-hour pre-onset window, whereas non-sepsis episodes were evaluated over the full observed stay.

Consequently, episode durations may be truncated at the boundaries of the sepsis warning window.

The resulting episode statistics should be interpreted descriptively rather than as a direct controlled comparison.

### 6.5 Threshold Selection

The threshold sensitivity experiment demonstrated that prediction thresholds have a substantial effect on both sepsis detection and false-alert burden.

However, evaluating several thresholds on the test set does not establish which threshold is clinically optimal.

The reference threshold of 0.748707 was used for consistency in the downstream warning analyses. Additional threshold optimization should be performed using validation data and a predefined selection strategy.

## 7. Future Work

The following directions could improve the research and extend the current experimental framework.

### 7.1 Investigation of ICU Timing Features

Further investigate the influence of ICULOS and HospAdmTime on model predictions.

This should include examining how timing variables relate to prediction availability, patient observation duration, and the recorded sepsis onset.

Additional experiments could evaluate models trained without selected timing features and compare their performance under clearly defined prediction settings.

### 7.2 Improved Temporal Deep Learning

Investigate alternative temporal architectures and sequence representations.

Possible directions include:

- Testing different LSTM sequence lengths.
- Evaluating alternative attention mechanisms.
- Investigating other temporal deep learning architectures.
- Exploring different feature combinations.
- Improving training strategies for imbalanced clinical time-series data.

### 7.3 Improved Missing-Value Handling

Explore alternative imputation methods and approaches that explicitly model missing clinical observations.

The objective would be to investigate whether improved missing-data representations lead to more reliable prediction performance.

### 7.4 Threshold Selection and Calibration

Investigate operating threshold selection using validation data.

Future experiments could consider:

- Validation-based threshold selection.
- Precision-recall trade-offs.
- Probability calibration.
- False-alert burden constraints.
- Detection performance at predefined warning windows.

The selected threshold should be evaluated on an independent test population without further tuning.

### 7.5 Improved Alert Analysis

Extend alert evaluation to investigate:

- Alert onset and termination.
- Duration of warning periods.
- Repeated alerts.
- Alert frequency per patient.
- Time between the first alert and recorded sepsis onset.
- Different warning-window definitions.

Future comparisons should use carefully defined and comparable observation periods.

### 7.6 External Validation

Evaluate the developed models using independent datasets from different hospitals or healthcare environments.

This would help determine whether the observed predictive performance is maintained across different patient populations and data collection settings.

### 7.7 Streamlit Application Development

Integrate the trained model and explainability components into the existing Streamlit application.

The application can provide an interface for:

- Uploading patient observation data.
- Validating input features.
- Generating model predictions.
- Displaying prediction probabilities.
- Presenting SHAP-based explanations.
- Demonstrating the experimental prediction workflow.

The application should be treated as a research demonstration rather than a clinically validated decision-support system.

## 8. Summary

The experiments demonstrate the usefulness of combining conventional machine learning, temporal deep learning, explainability, statistical comparison, and early warning evaluation for investigating sepsis prediction.

However, important limitations remain regarding class imbalance, missing observations, model performance, temporal feature dependence, warning-window availability, threshold selection, and external generalizability.

Future work should prioritize investigating ICU timing features, improving temporal modelling, refining alert evaluation, validating thresholds independently, and testing the models on external datasets.

These improvements would provide a stronger foundation for further research into explainable early clinical deterioration prediction.