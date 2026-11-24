# 07. Early Warning and Alert Analysis

## 1. Introduction

Conventional classification metrics evaluate whether a model correctly identifies positive and negative observations. However, in early clinical deterioration prediction, the timing and continuity of predictions are also important.

A model may correctly identify a positive observation without providing sufficient warning before the recorded onset.

To investigate this aspect, additional early warning experiments were conducted using the Original XGBoost model.

The analysis included:

1. Exact-hour lead-time evaluation.
2. Continuous warning-window evaluation.
3. Non-sepsis alert burden analysis.
4. Alert episode analysis.
5. Prediction threshold sensitivity analysis.

These experiments used predictions generated from the full test partition.

## 2. Evaluation Setup

The Original XGBoost model was used for the early warning experiments.

The full test prediction dataset contained:

| Parameter | Value |
|---|---:|
| Total hourly observations | 66,737 |
| Total patients | 1,721 |
| Sepsis patients | 148 |
| Non-sepsis patients | 1,573 |
| Positive hourly sepsis labels | 1,418 |
| Reference prediction threshold | 0.748707 |

The trained Original XGBoost model and its saved preprocessing components were used to generate predictions.

No model retraining was performed for these analyses.

### 2.1 Sepsis Onset Definition

For each sepsis patient, the onset was defined as the first hourly observation with a positive `SepsisLabel`.

The ICU observation hour (`ICULOS`) was used to calculate the difference between an observation and the recorded sepsis onset.

For an observation at ICU hour \(t\), the lead time was calculated as:

\[
\text{Lead Time} =
\text{Sepsis Onset Hour} - t
\]

A positive prediction exceeding the reference threshold was considered an alert.

## 3. Exact-Hour Lead-Time Evaluation

### 3.1 Objective

The exact-hour lead-time analysis investigated whether the model generated an alert at specific time points before the recorded sepsis onset.

Three lead times were evaluated:

- 6 hours before onset.
- 4 hours before onset.
- 2 hours before onset.

Patients without observations at the corresponding lead time were classified as unavailable for that specific evaluation.

### 3.2 Results

| Lead Time | Total Sepsis Patients | Eligible Patients | Unavailable Patients | Detected Patients | Missed Eligible Patients |
|---|---:|---:|---:|---:|---:|
| 6 hours | 148 | 113 | 35 | 37 | 76 |
| 4 hours | 148 | 121 | 27 | 34 | 87 |
| 2 hours | 148 | 128 | 20 | 37 | 91 |

The detection rates were:

| Lead Time | Detection Among Eligible Patients | Detection Among All Sepsis Patients |
|---|---:|---:|
| 6 hours | 32.74% | 25.00% |
| 4 hours | 28.10% | 22.97% |
| 2 hours | 28.91% | 25.00% |

### 3.3 Interpretation

The model generated above-threshold predictions at the specified lead times for a subset of sepsis patients.

The results varied across the three lead times.

The number of eligible patients also differed because not every patient had an observation at each exact time point.

Therefore, detection rates among eligible patients and detection rates among all sepsis patients were reported separately.

## 4. Continuous Warning-Window Evaluation

### 4.1 Objective

Exact-hour evaluation considers only one observation at a particular lead time.

To examine warning behavior across a broader period, a continuous warning window covering 2–6 hours before the recorded sepsis onset was evaluated.

A sepsis patient was considered detected if at least one prediction within this window exceeded the reference threshold.

This analysis measured whether the model generated an alert at any point in the specified warning window.

### 4.2 Sepsis Detection Results

| Parameter | Result |
|---|---:|
| Total sepsis patients | 148 |
| Patients with window observations | 128 |
| Patients without window observations | 20 |
| Patients detected within the window | 45 |
| Patients not detected within the window | 103 |
| Detection rate among all sepsis patients | 30.41% |
| Detection rate among eligible patients | 35.16% |

The detection rate among all sepsis patients was calculated as:

\[
\frac{45}{148}\times 100 = 30.41\%
\]

The detection rate among eligible patients was calculated as:

\[
\frac{45}{128}\times 100 = 35.16\%
\]

Patients without observations in the warning window were retained in the overall patient count and were not classified as detected.

### 4.3 Earliest Warning Time

For each detected sepsis patient, the earliest above-threshold prediction within the 2–6-hour window was identified.

The results were:

| Metric | Result |
|---|---:|
| Mean earliest warning time | 5.53 hours |
| Median earliest warning time | 6 hours |
| Earliest warnings at 6 hours | 37 |
| Earliest warnings between 4 and 6 hours | 3 |
| Earliest warnings between 2 and 4 hours | 5 |

Most detected patients had their earliest alert at the six-hour boundary.

This indicates that some warning activity may have started earlier than the evaluated window. The analysis cannot determine the actual start time of these warnings beyond the selected six-hour boundary.

### 4.4 Interpretation

The continuous warning-window analysis provides a broader assessment of early prediction behavior than exact-hour evaluation.

The model generated at least one alert within the 2–6-hour pre-onset window for 45 of the 148 sepsis patients.

However, detection within this window does not necessarily indicate a continuous or sustained warning throughout the entire period.

## 5. Non-Sepsis Alert Burden

### 5.1 Objective

An early warning model should also be evaluated in terms of alerts generated for patients without a recorded positive sepsis label.

The non-sepsis alert analysis examined the full observed stays of the 1,573 non-sepsis patients.

An alert was defined as a prediction score greater than or equal to the reference threshold of 0.748707.

### 5.2 Results

| Parameter | Result |
|---|---:|
| Total non-sepsis patients | 1,573 |
| Patients with at least one alert | 108 |
| Patients without alerts | 1,465 |
| Patient-level alert rate | 6.87% |
| Total observed patient-hours | 58,032 |
| Total alert hours | 342 |
| Alert hours per 1,000 observed hours | 5.89 |

The patient-level alert rate was calculated as:

\[
\frac{108}{1573}\times 100 = 6.87\%
\]

The alert-hour rate was calculated as:

\[
\frac{342}{58032}\times 1000
= 5.89
\]

These values describe the observed alert burden among non-sepsis patients.

They should not automatically be interpreted as clinical false-alarm rates because the analysis is based on recorded sepsis labels and does not establish whether every alert corresponds to an absence of other clinically relevant deterioration.

## 6. Alert Episode Analysis

### 6.1 Objective

A sequence of consecutive positive predictions may provide different information from isolated alerts.

To investigate this behavior, consecutive positive predictions in consecutive ICU hours were grouped into alert episodes.

An alert episode was defined as a sequence of consecutive observations in which the prediction exceeded the selected threshold.

A missing ICU hour or a non-alert prediction terminated the episode.

### 6.2 Episode Classification

Alert episodes were classified into two categories:

- Isolated episode: An episode lasting exactly one hour.
- Sustained episode: An episode lasting at least two consecutive hours.

### 6.3 Results

The episode analysis produced the following results:

| Metric | Sepsis (2–6h Pre-Onset) | Non-Sepsis (Full Observation) |
|---|---:|---:|
| Patients | 148 | 1,573 |
| Observed patient-hours | 604 | 58,032 |
| Patients with alerts | 45 | 108 |
| Total alert episodes | 52 | 173 |
| Isolated episodes | 10 | 96 |
| Sustained episodes | 42 | 77 |
| Mean episode duration | 3.54 hours | 1.98 hours |
| Median episode duration | 5 hours | 1 hour |

### 6.4 Interpretation

Within the defined pre-onset window, alert episodes associated with sepsis patients tended to last longer than those observed among non-sepsis patients.

The sepsis group had 42 sustained episodes out of 52 total episodes, whereas the non-sepsis group had 77 sustained episodes out of 173 total episodes.

However, these groups were evaluated over different observation periods.

The sepsis analysis was restricted to the 2–6-hour pre-onset window, while the non-sepsis analysis used the full observed stay.

Consequently:

- Episode durations in the sepsis group may be truncated at the warning-window boundaries.
- The total observed patient-hours differ substantially.
- Episode rates and durations should not be interpreted as a direct controlled comparison.

The results provide descriptive information about alert patterns rather than proof that sustained alerts are specific to sepsis.

## 7. Threshold Sensitivity Analysis

### 7.1 Objective

Prediction thresholds influence the number of observations classified as positive.

A lower threshold may increase detection but can also increase the number of alerts among non-sepsis patients.

A higher threshold may reduce alert burden while missing more sepsis patients.

To investigate this relationship, five thresholds were evaluated using the same full test prediction dataset.

### 7.2 Results

| Threshold | Detected Sepsis Patients | Detection Rate | Non-Sepsis Patients with Alerts | Patient-Level Alert Rate | Alert Hours per 1,000 Hours |
|---:|---:|---:|---:|---:|---:|
| 0.65 | 58 | 39.19% | 237 | 15.07% | 19.06 |
| 0.70 | 53 | 35.81% | 165 | 10.49% | 11.24 |
| 0.748707 | 45 | 30.41% | 108 | 6.87% | 5.89 |
| 0.80 | 40 | 27.03% | 59 | 3.75% | 2.55 |
| 0.85 | 30 | 20.27% | 19 | 1.21% | 0.50 |

The sepsis detection rates were calculated using all 148 sepsis patients. The same 128 patients had observations available in the 2–6-hour warning window at every evaluated threshold.

### 7.3 Interpretation

The results demonstrate a clear trade-off between sepsis detection and non-sepsis alert burden.

At a threshold of 0.65, the model detected 58 sepsis patients, corresponding to 39.19% of all sepsis patients. However, 237 non-sepsis patients received at least one alert.

At a threshold of 0.85, the model detected 30 sepsis patients, while only 19 non-sepsis patients received at least one alert.

At the reference threshold of 0.748707, the model detected 45 sepsis patients, and 108 non-sepsis patients received alerts.

These findings demonstrate that threshold selection has a substantial effect on early warning behavior.

The analysis was performed on the test set to examine sensitivity to threshold changes. It was not used to select an optimal operating threshold.

## 8. Threshold Consistency

Two different Original XGBoost threshold values were recorded during the experimental workflow:

- Saved model artifact threshold: 0.734528.
- Common six-model comparison threshold: 0.748706579208374.

The downstream early warning experiments used 0.748707 to remain consistent with the common model comparison threshold.

This distinction is documented to preserve reproducibility and prevent confusion between the threshold stored in the model artifact and the threshold used in the downstream warning analyses.

## 9. Limitations

The early warning experiments have several limitations:

1. The onset definition is based on the first recorded positive SepsisLabel and does not necessarily represent the exact clinical onset of sepsis.
2. Some patients did not have observations available at the specified lead times or within the continuous warning window.
3. The continuous warning analysis identifies at least one alert within the window and does not establish continuous warning throughout the period.
4. Warnings at the six-hour boundary may have started earlier than the evaluated window.
5. Non-sepsis patients were evaluated over their full observed stays, whereas sepsis patients were evaluated within a restricted pre-onset window for episode analysis.
6. Alert episodes may be truncated by the boundaries of the evaluated warning window.
7. The threshold sensitivity analysis does not establish a clinically optimal operating threshold.
8. Retrospective test-set performance does not establish clinical effectiveness or readiness for deployment.

## 10. Summary

The early warning experiments extended the model evaluation beyond conventional hourly classification metrics.

The Original XGBoost model detected 45 of 148 sepsis patients within the 2–6-hour pre-onset window at the reference threshold of 0.748707.

The non-sepsis analysis identified 108 patients with at least one alert, corresponding to 6.87% of the non-sepsis population.

Alert episode analysis provided additional information about the duration and continuity of positive predictions.

Threshold sensitivity analysis demonstrated that lower thresholds increased sepsis detection while also increasing non-sepsis alert burden.

Overall, these experiments provide a retrospective assessment of early warning behavior and identify important considerations for future development and validation.