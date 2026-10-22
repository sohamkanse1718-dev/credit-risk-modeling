# Lending Club Default Prediction — Business Decisions

## Business Objective
Identify high-risk loans by detecting as many actual defaulters as possible while controlling good borrowers incorrectly flagged.

## Modeling Decisions
- Compared XGBoost, regularized XGBoost, and Logistic Regression.
- Used time-ordered cross-validation for model tuning.
- Selected the model and threshold using validation data.
- Evaluated the frozen selection on the held-out test set.
- Minimized expected cost subject to at least 50% default recall.

## Cost Assumption
Experimented with cost ratios of **1:1, 2:1, 3:1, 4:1, 5:1, and 8:1** to study how business priorities affect threshold selection and the precision–recall trade-off.

Selected **4:1** as the working assumption: a missed defaulter (false negative) costs four times as much as a good borrower incorrectly flagged (false positive).

Expected cost per loan:

`(FP + 4 × FN) / N`

The ratio is an assumption requiring validation against actual business costs.

## Final Selection
- **Model:** Regularized XGBoost
- **Threshold:** 0.19
- **Test ROC-AUC:** 0.6983
- **Test PR-AUC:** 0.3358
- **Default recall:** 58.88%
- **Default precision:** 30.20%
- **Expected test cost per loan:** 0.5545
- **Approve-all baseline cost per loan:** 0.7380

## Business Interpretation
The selected model achieves lower expected cost than the approve-all baseline under the assumed cost ratio. However, false positives remain substantial, so predictions should support risk assessment rather than automatically determine loan approval or rejection.

## Limitations
- Moderate model discrimination and substantial false positives.
- Different validation and test default rates may indicate temporal distribution shift.
- Cost ratio is assumed, not derived from verified lender losses.
- Monitor performance and threshold stability on future loan cohorts.

## Deployment Artifact
`models/model.pkl` — saved model pipeline and decision threshold. Verify that the artifact loads correctly and reproduces expected predictions before deployment.
