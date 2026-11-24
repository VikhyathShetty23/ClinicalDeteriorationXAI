# 08. Statistical Analysis

## 1. Introduction

Comparing machine learning models using a single test-set performance estimate does not fully describe the uncertainty associated with the observed differences.

To investigate the reliability of the performance comparisons, a paired patient-level bootstrap analysis was conducted.

The analysis focused on comparing the Original XGBoost model against the other five model configurations using AUROC and AUPRC.

## 2. Objective

The primary objectives of the statistical analysis were:

1. To estimate differences in AUROC between Original XGBoost and the other models.
2. To estimate differences in AUPRC between Original XGBoost and the other models.
3. To calculate 95% confidence intervals for the observed metric differences.
4. To investigate whether the confidence intervals included zero.

## 3. Paired Patient-Level Bootstrap

### 3.1 Bootstrap Procedure

A paired patient-level bootstrap procedure was performed using the common test population.

The test population contained 1,721 unique patients.

The procedure involved:

1. Randomly sampling 1,721 patients with replacement.
2. Retaining the hourly observations belonging to the sampled patients.
3. Accounting for repeated selections of the same patient through patient multiplicity weights.
4. Calculating AUROC and AUPRC for each model on the resampled population.
5. Calculating the performance difference between Original XGBoost and the comparison model.
6. Repeating the procedure 2,000 times.

The same patient resamples were used for each model pair to preserve the paired comparison.

### 3.2 Performance Difference

For each bootstrap resample, the performance difference was calculated as:

\[
\Delta M =
M_{\text{Original XGBoost}}
-
M_{\text{Comparison Model}}
\]

where \(M\) represents either AUROC or AUPRC.

A positive difference indicates a higher metric value for Original XGBoost.

## 4. Bootstrap Results

The following table presents the observed performance differences and their corresponding 95% confidence intervals.

### 4.1 AUROC Comparison

| Comparison | AUROC Difference | 95% Confidence Interval |
|---|---:|---:|
| Original XGBoost vs Random Forest | 0.02133 | [-0.00447, 0.04681] |
| Original XGBoost vs Logistic Regression | 0.01295 | [-0.02018, 0.04742] |
| Original XGBoost vs Linear SVM | 0.01194 | [-0.02086, 0.04595] |
| Original XGBoost vs Clinical-only XGBoost | 0.08574 | [0.05034, 0.11955] |
| Original XGBoost vs Temporal Attention LSTM | 0.10512 | [0.06277, 0.14758] |

### 4.2 AUPRC Comparison

| Comparison | AUPRC Difference | 95% Confidence Interval |
|---|---:|---:|
| Original XGBoost vs Random Forest | 0.00984 | [-0.00167, 0.03329] |
| Original XGBoost vs Logistic Regression | -0.00353 | [-0.02775, 0.02428] |
| Original XGBoost vs Linear SVM | -0.00208 | [-0.02532, 0.02692] |
| Original XGBoost vs Clinical-only XGBoost | 0.03733 | [0.01782, 0.07009] |
| Original XGBoost vs Temporal Attention LSTM | 0.04359 | [0.02162, 0.07560] |

## 5. Interpretation of AUROC Results

The confidence intervals for comparisons against Random Forest, Logistic Regression, and Linear SVM included zero.

Therefore, the bootstrap analysis did not establish statistically distinguishable AUROC differences between Original XGBoost and these three models using the calculated intervals.

For Clinical-only XGBoost and Temporal Attention LSTM, the confidence intervals were entirely positive.

This indicates that Original XGBoost had higher AUROC than these two configurations across the evaluated bootstrap comparisons, with confidence intervals excluding zero.

## 6. Interpretation of AUPRC Results

The AUPRC confidence intervals for Random Forest, Logistic Regression, and Linear SVM also included zero.

Therefore, the analysis did not establish statistically distinguishable AUPRC differences for these comparisons.

For Clinical-only XGBoost and Temporal Attention LSTM, the confidence intervals were entirely positive.

This indicates higher AUPRC for Original XGBoost compared with these two configurations under the paired bootstrap analysis.

## 7. Summary of Bootstrap Findings

| Comparison | AUROC CI Excludes Zero | AUPRC CI Excludes Zero |
|---|---|---|
| Original XGBoost vs Random Forest | No | No |
| Original XGBoost vs Logistic Regression | No | No |
| Original XGBoost vs Linear SVM | No | No |
| Original XGBoost vs Clinical-only XGBoost | Yes | Yes |
| Original XGBoost vs Temporal Attention LSTM | Yes | Yes |

The results indicate that the performance differences between Original XGBoost and the three stronger competing tabular configurations were not clearly distinguishable using these bootstrap confidence intervals.

However, Original XGBoost showed higher AUROC and AUPRC than Clinical-only XGBoost and Temporal Attention LSTM, with confidence intervals excluding zero.

## 8. Statistical Considerations and Limitations

The bootstrap analysis has several considerations:

1. The resampling unit was the patient rather than the individual hourly observation, accounting for the grouping of repeated observations within patients.
2. The same patient resamples were used across model comparisons to maintain pairing.
3. The analysis used 2,000 bootstrap resamples.
4. The confidence intervals were not adjusted for multiple comparisons.
5. Confidence intervals including zero indicate that the analysis did not establish a difference using this procedure; they do not prove that the models are equivalent.
6. The bootstrap analysis reflects uncertainty under the evaluated test population and does not establish generalizability to other hospitals or patient populations.

## 9. Summary

A paired patient-level bootstrap analysis was performed to investigate differences between Original XGBoost and the other five model configurations.

The results showed that the AUROC and AUPRC differences between Original XGBoost and Random Forest, Logistic Regression, and Linear SVM were not clearly distinguishable using the calculated 95% confidence intervals.

In contrast, Original XGBoost demonstrated higher AUROC and AUPRC than Clinical-only XGBoost and Temporal Attention LSTM, with confidence intervals excluding zero.

The analysis provides additional statistical context for interpreting the model comparison results.