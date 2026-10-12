# Changelog

## 3.0.0 - 2026-10-12 - Multi-tenant platform, integrations and experimentation

- Multi-tenant organisations with isolation enforced in the repository layer; SQLAlchemy storage
  (SQLite default, PostgreSQL via `RETENTIONIQ_DATABASE_URL`).
- Sign-in with PBKDF2 password hashing, account lockout and idle timeout; server-side RBAC;
  hashed, org-scoped API keys; users & API keys admin page. Simple local demo logins (`demo123`).
- Database-backed job queue, worker and scheduler: background batch scoring, scheduled
  rescoring, folder-drop imports, monitoring snapshots, retraining jobs; Operations page.
- Integrations: transactional outbox, HMAC-signed webhooks with idempotency keys, retries with
  backoff, dead-letter replay, SSRF protection, sandbox CRM; Salesforce/HubSpot CSV exports.
- Score history and rising-risk detection with `customer.high_risk` / `customer.risk_rising` events.
- Experiments: registry, power analysis, two-proportion tests, held-out control arms; uplift
  T-learner with a simulation lab and Qini evaluation.
- Model lifecycle: candidate → staging → production with approval reasons and rollback;
  champion/challenger shadow scoring; retraining with tenant feedback labels and optional tuning.
- Analytics assistant over a governed semantic layer (restricted query spec, no generated SQL).
- API: rate limiting, Prometheus `/metrics`, security headers, new job/webhook/experiment/
  assistant/model endpoints. 126 tests.

## 2.0.0 - 2026-10-11 - RetentionIQ platform

Rebuild of the churn app into a retention decision-support platform, following
the Industry Blueprint build order.

### Phase 0 - stabilise baseline
- Re-validated historical metrics: the previous model measured ROC-AUC 0.834,
  PR-AUC 0.642, precision 0.52 on a held-out 20% split (the old notes reported
  78.57% precision).
- Fixed: `xgboost` missing from requirements; single-prediction form sent
  categories the model never saw; `src/predict.py` loaded files that did not exist;
  `app.py` was a debug stub.
- Pinned Python 3.10-3.12 and all dependencies.

### Phase 1 - production-quality ML
- Data contract with row-level validation; reproducible `make train`; train /
  validation / test split; logistic-regression baseline vs RF vs XGBoost; cost-based
  threshold; bootstrap CIs; calibration; lift; subgroup checks; release gate;
  versioned registry with SHA-256 integrity; generated model card + evaluation report.

### Phase 2 - product workflow
- Risk work queue, customer detail, batch upload wizard with error reports and
  exports, segments with sample-size guardrails.

### Phase 3 - API
- FastAPI `/health`, `/v1/model-info`, `/v1/predictions`, `/v1/predictions/batch`,
  customer summary, recommendations, outcomes, monitoring; optional API key;
  request IDs and structured logs.

### Phase 4 - explainability & trust
- Per-customer drivers (TreeSHAP / linear), plain-language reason codes, driver
  stability check, global importance, trust panel, what-if sensitivity.

### Phase 5 - monitoring & governance
- PSI drift by window or batch job, data-quality and score alerts with runbooks,
  delayed-label performance, audit log, model promotion/rollback.

### Phase 6 (+ start of 8) - decision intelligence
- Policy-based next-best action with eligibility, cost, approval and expected
  value; action lifecycle with outcomes; deterministic control group and
  impact-by-arm reporting.
