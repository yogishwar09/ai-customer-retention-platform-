# Operational runbooks

Each alert in **Monitoring** links here. Owners are roles, not people - map them
to your team.

## Data drift
**Signal:** PSI ≥ 0.10 (warning) or ≥ 0.25 (alert) on a feature or on the score
distribution, for the selected window, versus the training reference profile.
**Owner:** ML engineer.

1. Open Monitoring → Data drift and switch the window (1/7/30 days, or a single
   batch job) to localise *when* the shift started.
2. Check whether the shift comes from one upload (Monitoring → Volume & quality
   → Batch jobs). A single unusual file is a data-source question, not a model one.
3. Compare reference vs current summaries in the drift table. Ask the data owner
   whether a business change explains it (new pricing, campaign, product launch).
4. If the population genuinely changed, check delayed-label performance before
   acting. **Do not retrain automatically** - retrain with `make train`, review the
   evaluation report and release gate, then promote in Governance.

## Data quality
**Signal:** share of predictions with data-quality warnings above
`monitoring.dq_warning_rate_alert`. **Owner:** data analyst.

1. Download the batch error report / warnings from Batch scoring.
2. Most common causes: blank `TotalCharges`, totals inconsistent with tenure ×
   monthly charges, source-system export changes.
3. Fix at source; rescore affected customers. Records with warnings are routed to
   the `DATA_REVIEW` playbook instead of customer outreach.

## Scores
**Signal:** all scores identical, or a sudden mean-score jump with no drift.
**Owner:** ML engineer.

1. Check `/health` and the trust panel - is the expected model version loaded?
2. Run `make test` - the train/inference consistency test catches pipeline breaks.
3. Roll back: Governance → Model registry → promote the previous version
   (or `python scripts/registry_cli.py promote <version>`).

## Performance
**Signal:** delayed-label PR-AUC more than 0.10 below the test baseline.
**Owner:** ML engineer.

1. Confirm label quality and volume (labels must post-date the score).
2. Look at segment-level performance in Model lab and Segments.
3. Retrain on recent labelled data, compare in the evaluation report, promote only
   if the release gate passes and subgroup gaps are acceptable.

## Batch jobs
**Signal:** failed batch job. **Owner:** platform admin.

1. Governance → Audit log shows the `batch.failed` event and error message.
2. Re-run with the downloaded error report fixed. Files over the configured size
   or row limits are rejected before scoring.

## Rollback (app or model)
- **Model:** promote the previous version (Governance or registry CLI). Every
  score stores its `model_version`, so history stays traceable.
- **App:** redeploy the previous tagged release; the SQLite/Postgres schema is
  additive, so older app versions can read it.
