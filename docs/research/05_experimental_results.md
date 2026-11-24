# 05. Experimental Results

## 1. Introduction

This section presents the experimental results obtained from evaluating six machine learning configurations for early sepsis prediction using the PhysioNet 2019 Sepsis Challenge dataset.

The models were evaluated using a common test population to ensure consistency in the comparison. Performance was measured using AUROC, AUPRC, precision, recall, and F1-score.

Additional experiments investigated model performance through feature ablation, early warning evaluation, and threshold sensitivity analysis.

## 2. Experimental Evaluation Setup

The common model comparison was conducted using:

| Parameter | Value |
|---|---:|
| Test patients | 1,721 |
| Common test hourly endpoints | 58,132 |
| Positive hourly labels | 1,300 |
| Positive label prevalence | 2.24% |
| Number of model configurations | 6 |

The full test partition contained 66,737 hourly observations. The common comparison population contained 58,132 eligible endpoints shared across the six model configurations.

This distinction is important because the model comparison and subsequent full-test early warning analysis used different evaluation populations.

## 3. Six-Model Performance Comparison

The following table presents the performance of all six model configurations on the common test population.

| Model | AUROC | AUPRC | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|---:|
| Original XGBoost | **0.7912** | 0.0994 | 0.1413 | 0.3377 | **0.1993** |
| Random Forest | 0.7699 | 0.0896 | 0.1327 | **0.3600** | 0.1939 |
| Logistic Regression | 0.7783 | **0.1030** | 0.1334 | 0.3385 | 0.1914 |
| Linear SVM | 0.7793 | 0.1015 | 0.1312 | 0.3254 | 0.1870 |
| Clinical-only XGBoost | 0.7055 | 0.0621 | 0.0856 | 0.2200 | 0.1232 |
| Temporal Attention LSTM | 0.6861 | 0.0559 | 0.0789 | 0.1738 | 0.1086 |

### 3.1 Observations

The experimental results indicate the following:

- Original XGBoost achieved the highest AUROC of 0.7912.
- Original XGBoost achieved the highest F1-score of 0.1993.
- Logistic Regression achieved the highest AUPRC of 0.1030.
- Random Forest achieved the highest recall of 0.3600.
- Linear SVM produced performance relatively close to Logistic Regression and Original XGBoost.
- Clinical-only XGBoost showed lower predictive performance than the original configuration.
- Temporal Attention LSTM achieved the lowest AUROC and AUPRC among the six evaluated configurations.

Overall, Original XGBoost provided the strongest AUROC and F1-score, while Logistic Regression achieved the highest AUPRC.

## 4. Confusion Matrix Results

The confusion matrices obtained on the common test population are presented below.

| Model | True Negative | False Positive | False Negative | True Positive |
|---|---:|---:|---:|---:|
| Original XGBoost | 54,165 | 2,667 | 861 | 439 |
| Random Forest | 53,772 | 3,060 | 832 | 468 |
| Logistic Regression | 53,974 | 2,858 | 860 | 440 |
| Linear SVM | 54,032 | 2,800 | 877 | 423 |
| Clinical-only XGBoost | 53,775 | 3,057 | 1,014 | 286 |
| Temporal Attention LSTM | 54,195 | 2,637 | 1,074 | 226 |

The confusion matrices illustrate the differences in positive-class detection and false-positive predictions across the evaluated models.

Random Forest identified the largest number of positive hourly observations, while Original XGBoost achieved the highest F1-score.

## 5. Feature Ablation Results

Feature ablation experiments were conducted to investigate the contribution of ICU and hospital timing information.

The Original XGBoost model was compared with three modified configurations.

| Configuration | AUROC | AUPRC | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|---:|
| Original XGBoost | 0.7912 | 0.0994 | 0.1413 | 0.3377 | 0.1993 |
| Without ICULOS | 0.7167 | 0.0613 | 0.0779 | 0.2285 | 0.1162 |
| Without HospAdmTime | 0.7910 | 0.1035 | 0.1524 | 0.3138 | 0.2051 |
| Without both | 0.7055 | 0.0621 | 0.0827 | 0.2315 | 0.1218 |

### 5.1 Observations

Removing ICULOS resulted in a substantial reduction in model performance.

AUROC decreased from 0.7912 to 0.7167, while AUPRC decreased from 0.0994 to 0.0613.

Removing HospAdmTime alone had little effect on AUROC and produced a slight increase in AUPRC and F1-score.

Removing both timing features resulted in further performance reduction.

These findings suggest that ICULOS provides important predictive information within the Original XGBoost model.

## 6. Early Warning Evaluation

The early warning experiments were conducted using the Original XGBoost model on the full test population of 66,737 hourly observations from 1,721 patients.

The reference prediction threshold was 0.748707.

### 6.1 Exact-Hour Lead-Time Analysis

Sepsis onset was defined as the first recorded hourly observation with a positive SepsisLabel for each sepsis patient.

The model was evaluated at 6-hour, 4-hour, and 2-hour lead times before the recorded onset.

| Lead Time | Eligible Patients | Detected Patients | Missed Eligible Patients | Detection Among All Sepsis Patients |
|---|---:|---:|---:|---:|
| 6 hours | 113 | 37 | 76 | 25.00% |
| 4 hours | 121 | 34 | 87 | 22.97% |
| 2 hours | 128 | 37 | 91 | 25.00% |

Some patients did not have observations available at the specified lead time. These patients were classified as unavailable rather than missed.

### 6.2 Continuous Warning-Window Analysis

A continuous warning window covering 2–6 hours before recorded sepsis onset was evaluated.

The results were:

| Parameter | Result |
|---|---:|
| Total sepsis patients | 148 |
| Patients with window observations | 128 |
| Patients without window observations | 20 |
| Patients detected in the window | 45 |
| Patients not detected in the window | 103 |
| Detection rate among all sepsis patients | 30.41% |
| Detection rate among eligible patients | 35.16% |
| Mean earliest warning time | 5.53 hours |
| Median earliest warning time | 6 hours |

The earliest warning distribution among the 45 detected patients was:

- 37 patients had their earliest warning at 6 hours.
- 3 patients had their earliest warning between 4 and 6 hours.
- 5 patients had their earliest warning between 2 and 4 hours.

The analysis demonstrates that the model generated at least one above-threshold prediction within the evaluated warning window for 45 sepsis patients.

The six-hour boundary concentration indicates that some warnings may have started earlier than the evaluated window.

### 6.3 Non-Sepsis Alert Burden

The non-sepsis evaluation included 1,573 patients and 58,032 observed patient-hours.

| Parameter | Result |
|---|---:|
| Non-sepsis patients | 1,573 |
| Patients receiving at least one alert | 108 |
| Patient-level alert rate | 6.87% |
| Total alert hours | 342 |
| Total observed hours | 58,032 |
| Alert hours per 1,000 observed hours | 5.89 |

These results provide an estimate of the alert burden among patients without a recorded positive sepsis label.

## 7. Alert Episode Analysis

Consecutive positive predictions in consecutive ICU hours were grouped into alert episodes.

An episode lasting one hour was classified as isolated, while an episode lasting at least two consecutive hours was classified as sustained.

The results were:

| Metric | Sepsis (2–6h pre-onset) | Non-Sepsis (Full Observation) |
|---|---:|---:|
| Patients | 148 | 1,573 |
| Observed patient-hours | 604 | 58,032 |
| Patients with alerts | 45 | 108 |
| Total episodes | 52 | 173 |
| Isolated episodes | 10 | 96 |
| Sustained episodes | 42 | 77 |
| Mean episode duration | 3.54 hours | 1.98 hours |
| Median episode duration | 5 hours | 1 hour |

Within the defined pre-onset window, sepsis-associated alert episodes tended to be longer than those observed among non-sepsis patients.

However, the observation periods differ between the two groups. Sepsis episodes were restricted to the 2–6-hour pre-onset window, whereas non-sepsis episodes were evaluated across the full observed stay.

Episode durations may also be truncated by the boundaries of the sepsis warning window. Therefore, the episode durations should not be interpreted as a direct controlled comparison between the groups.

## 8. Threshold Sensitivity Analysis

A threshold sensitivity analysis was performed to investigate the relationship between sepsis detection and false-alert burden.

Five prediction thresholds were evaluated.

| Threshold | Detected Sepsis Patients | Detection Rate Among All Sepsis Patients | Non-Sepsis Patients with Alerts | Patient-Level Alert Rate | Alert Hours per 1,000 Hours |
|---:|---:|---:|---:|---:|---:|
| 0.65 | 58 | 39.19% | 237 | 15.07% | 19.06 |
| 0.70 | 53 | 35.81% | 165 | 10.49% | 11.24 |
| 0.748707 | 45 | 30.41% | 108 | 6.87% | 5.89 |
| 0.80 | 40 | 27.03% | 59 | 3.75% | 2.55 |
| 0.85 | 30 | 20.27% | 19 | 1.21% | 0.50 |

### 8.1 Observations

Lower prediction thresholds increased the number of detected sepsis patients but also increased the number of non-sepsis patients receiving alerts.

At a threshold of 0.65, the model detected 58 sepsis patients, corresponding to 39.19% of all sepsis patients. However, 237 non-sepsis patients received at least one alert.

At a threshold of 0.85, only 30 sepsis patients were detected, while the number of non-sepsis patients receiving alerts decreased to 19.

At the reference threshold of 0.748707, the model detected 45 sepsis patients, with 108 non-sepsis patients receiving at least one alert.

These findings demonstrate the trade-off between detection sensitivity and false-alert burden.

The sensitivity experiment was conducted as a robustness analysis and was not used to select an optimal threshold from the test data.

## 9. Overall Experimental Findings

The experiments produced several important findings:

1. Original XGBoost achieved the highest AUROC and F1-score among the six evaluated configurations.
2. Logistic Regression achieved the highest AUPRC.
3. Random Forest achieved the highest recall.
4. Removing ICULOS substantially reduced XGBoost performance.
5. The Temporal Attention LSTM underperformed the evaluated tabular models in this experimental setup.
6. The Original XGBoost model detected 45 of 148 sepsis patients within the 2–6-hour warning window.
7. Alert episode and threshold sensitivity analyses demonstrated the importance of examining warning behavior beyond conventional classification metrics.
8. Lower prediction thresholds increased detection while also increasing the false-alert burden.

## 10. Summary

The experimental results show that the Original XGBoost configuration provided the strongest overall AUROC and F1-score among the evaluated models.

The additional experiments provided insights into feature importance, model behavior, early warning detection, alert episodes, and threshold sensitivity.

Although the findings demonstrate the potential usefulness of machine learning for retrospective sepsis prediction, the observed performance and alert behavior do not establish clinical readiness. Further validation and investigation of temporal features, threshold selection, and generalizability are required.