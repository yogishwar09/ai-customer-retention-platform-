# Model card - `churn-xgboost-20261011090305`

## Intended use
Rank existing telecom subscribers by **estimated churn risk** so a retention team can
prioritise *advisory* outreach. Scores support a human decision; they do not decide.

## Out-of-scope / prohibited use
- Denying service, raising prices, or any punitive treatment based on a score.
- Use on populations or products materially different from the training data
  (non-telecom, different market) without re-validation.
- Treating feature contributions as causal reasons why a customer will leave.

## Model
- Algorithm: **xgboost** inside a scikit-learn `Pipeline` (shared preprocessing).
- Operating threshold: **0.29**, chosen on validation data by
  minimising expected cost with FN:FP = 3.0:1.0.
- Explanation method: TreeSHAP (XGBoost native), log-odds.
- Schema version: 1.0.0; trained 2026-10-11T09:03:05+00:00.

## Training data
IBM Telco Customer Churn sample (7,043 customers, churn prevalence
26.5%). Churn = 'Yes' in the IBM Telco sample: customer left within the last month of the snapshot. No timestamps are available, so the split is stratified random rather than time-based.

## Performance (held-out test set, n=1,409)
- ROC-AUC 0.843 (95% CI 0.819-0.865)
- PR-AUC 0.654 (95% CI 0.603-0.704); baseline (random) = prevalence 26.5%
- Brier score 0.137
- At the operating threshold: precision 0.529, recall 0.775

Most influential inputs (mean |contribution|): Contract, Tenure (months), Monthly charges, Online security, Internet service.

## Fairness & limitations
- Subgroup metrics for gender and senior-citizen status are reported in
  `reports/evaluation_report.md`; differences should be reviewed before deployment.
  `gender` is kept as an input only to enable this audit; consider removing it for targeting.
- Snapshot data without timestamps: no time-based validation, so real-world performance
  may be lower than reported.
- No intervention data exists: the model predicts churn, it does not estimate which
  action would prevent it. Expected-value figures in the app use assumed effectiveness.

## Release decision
Release gate **passed** →
registered as `production`.
