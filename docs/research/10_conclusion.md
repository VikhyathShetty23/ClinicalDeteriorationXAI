# 10. Conclusion

## 1. Conclusion

This research investigated the application of explainable machine learning and temporal deep learning techniques for early sepsis prediction using physiological and contextual information from the PhysioNet 2019 Sepsis Challenge dataset.

The study followed a structured experimental workflow involving dataset preparation, patient-level splitting, feature engineering, model development, performance evaluation, explainability, feature ablation, statistical comparison, and early warning analysis.

Six model configurations were developed and evaluated: Original XGBoost, Clinical-only XGBoost, Random Forest, Logistic Regression, Linear SVM, and Temporal Attention LSTM.

Among the evaluated configurations, Original XGBoost achieved the highest AUROC of 0.7912 and F1-score of 0.1993 on the common test population. Logistic Regression achieved the highest AUPRC of 0.1030, while Random Forest achieved the highest recall of 0.3600.

The results demonstrated that different algorithms exhibited different strengths when evaluated using multiple classification metrics. The paired patient-level bootstrap analysis further showed that performance differences between Original XGBoost and Random Forest, Logistic Regression, and Linear SVM were not clearly distinguishable using the calculated 95% confidence intervals. However, Original XGBoost showed higher AUROC and AUPRC than Clinical-only XGBoost and Temporal Attention LSTM, with confidence intervals excluding zero.

SHAP explainability identified ICULOS, HospAdmTime, and Age as the most influential features in the Original XGBoost model. Feature ablation experiments further demonstrated the importance of ICULOS, as removing this feature resulted in a substantial reduction in predictive performance.

The early warning experiments extended the evaluation beyond conventional classification metrics. At the reference threshold of 0.748707, the Original XGBoost model detected 45 of 148 sepsis patients within the 2–6-hour pre-onset warning window, corresponding to a detection rate of 30.41% among all sepsis patients.

The non-sepsis analysis identified 108 patients with at least one alert, representing 6.87% of the non-sepsis population. Alert episode analysis provided additional information about the duration and continuity of positive predictions within the defined observation windows.

Threshold sensitivity analysis demonstrated a clear relationship between prediction threshold, sepsis detection, and false-alert burden. Lower thresholds increased detection but also increased alerts among non-sepsis patients, whereas higher thresholds reduced alert burden while decreasing detection.

Overall, the research demonstrates the value of combining predictive modelling, explainability, feature ablation, statistical comparison, and early warning analysis when investigating machine learning approaches for sepsis prediction.

However, the experiments also identified important limitations, including class imbalance, missing clinical measurements, relatively low positive-class precision, dependence on ICU timing information, and restrictions associated with retrospective evaluation.

The findings should therefore be interpreted as experimental results rather than evidence of clinical effectiveness or deployment readiness.

## 2. Research Contributions

The main contributions of this research are:

1. **Comparative model evaluation:** A systematic comparison of six machine learning configurations using a common test population.
2. **Temporal modelling:** Investigation of an LSTM architecture incorporating an attention mechanism for sequential physiological observations.
3. **Explainable AI:** Application of SHAP to investigate the contribution of physiological, temporal, and contextual features.
4. **Feature ablation:** Investigation of the predictive contribution of ICU and hospital timing variables.
5. **Statistical comparison:** Patient-level paired bootstrap analysis to examine differences in AUROC and AUPRC.
6. **Early warning evaluation:** Investigation of exact-hour lead times, continuous warning windows, and alert episodes.
7. **Threshold sensitivity:** Analysis of the trade-off between early detection and non-sepsis alert burden.

## 3. Future Direction

Further research should investigate the contribution of ICU timing variables, improve temporal deep learning performance, explore alternative missing-value handling strategies, and refine prediction threshold selection using validation data.

External validation using independent clinical datasets is also necessary to assess the generalizability of the developed models.

The existing Streamlit application can serve as a research demonstration interface for exploring patient observations, generating predictions, and displaying model explanations. However, clinical validation and appropriate safety evaluation would be required before any real-world clinical use.

## 4. Final Remarks

This research provides an experimental foundation for exploring explainable machine learning approaches to early clinical deterioration detection using ICU time-series data.

By combining model comparison, feature-level explanations, statistical analysis, and early warning evaluation, the study offers a broader understanding of predictive model behavior and its limitations.

The findings highlight both the potential and the challenges of applying machine learning to retrospective clinical prediction problems and provide directions for continued research and improvement.