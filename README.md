# RetentionIQ - AI Customer Retention Platform

**Identify risk early → explain why → recommend the next best action → prove whether it worked.**

RetentionIQ turns a telecom churn model into a multi-tenant retention decision platform:
validated data in, explainable risk scores out, policy-based advisory actions, randomised
experiments to measure impact, signed CRM webhooks, background jobs, drift and performance
monitoring, governed model releases, and a safe natural-language analytics assistant. The
Streamlit app, the REST API and the background worker all share one prediction service.

> Built on the public IBM Telco Customer Churn sample (7,043 customers). Model metrics are
> reproduced by `make train` on a held-out test set. Revenue and expected-value figures are
> labelled estimates, and model accuracy is never presented as evidence of retention impact.

---

## Quick start

Requires Python 3.10-3.12 (3.11 recommended).

```bash
make install
```

```bash
make seed
```

```bash
make app
```

Open http://localhost:8501 and sign in with a demo account:

| Username | Password | Access |
|---|---|---|
| `admin` | `demo123` | Demo Telco - every page, including Integrations and Users & API keys |
| `manager` | `demo123` | Approvals, exports, experiments |
| `agent` | `demo123` | Score customers, log actions and outcomes |
| `viewer` | `demo123` | Read-only |
| `northwind` | `demo123` | A second, fully isolated organisation (admin) |

Demo accounts are for local use only - the seeder refuses to create them when
`RETENTIONIQ_ENV=production`. The demo API key is written to `artifacts/demo_credentials.txt`.

REST API (docs at http://localhost:8000/docs) and a dedicated background worker:

```bash
make api
```

```bash
make worker
```

The app also runs an in-process worker by default (`RETENTIONIQ_INPROCESS_WORKER=1`), and on a
fresh host it trains a model and seeds the demo automatically on first start.

**Streamlit Community Cloud:** main file `app.py`, and choose **Python 3.11** in the app settings.

## What's inside

| Area | What you get |
|---|---|
| **Executive overview** | Customers scored, risk tiers, estimated revenue at risk, rising-risk count, action pipeline, data freshness |
| **Ask the data** | Plain-English questions answered from approved metrics only; small groups suppressed; every question audited. Questions are parsed into a restricted query spec - no generated SQL is ever run |
| **Risk work queue** | Prioritise by expected value, risk, trend or customer value; reason codes; rising-risk filter; Salesforce / HubSpot / generic CSV export |
| **Customer detail** | Score, drivers with stability check, score history, recommended playbook with assumptions, experiment arm, action timeline (create → approve → outcome) |
| **Score a customer** | Validated form, drivers, recommendation, what-if model sensitivity |
| **Batch scoring** | Column mapping, row-level validation and error reports, synchronous or background scoring, results export, per-file drift |
| **Segments & cohorts** | Observed churn with 95% intervals, minimum cohort size, contract × tenure heatmap |
| **Model lab** | Model comparison, ROC/PR, calibration, threshold trade-offs, lift, precision@K, subgroup gaps |
| **Experiments & uplift** | Experiment registry, power analysis, two-proportion tests with CIs, held-out control groups; uplift (T-learner) simulation lab with Qini curves |
| **Monitoring** | PSI drift by window or batch job, data-quality / score / integration alerts with runbooks, delayed-label performance |
| **Governance** | Model lifecycle (candidate → staging → production with approval reason, rollback), champion-vs-challenger shadow scoring, retraining with feedback labels, model cards, audit log |
| **Operations** | Job queue, schedules (webhook delivery, inbox imports, scheduled rescoring, monitoring snapshots), retries and dead jobs |
| **Integrations** | HMAC-signed webhooks with idempotency keys, exponential-backoff retries, dead-letter replay, SSRF protection; in-process sandbox CRM; folder-drop scheduled imports |
| **Users & API keys** | Organisation users and roles, one-time passwords, hashed API keys, permission matrix |

## Platform features

- **Multi-tenancy:** every business table carries `org_id`; isolation is enforced in the repository layer and tested.
- **Authentication & RBAC:** PBKDF2 password hashing, account lockout after repeated failures, idle session timeout; roles `viewer` < `agent` < `manager` < `admin` enforced server-side in the UI and API.
- **API:** FastAPI with typed contracts, Bearer API keys scoped to an organisation and role, per-key rate limiting, request IDs, structured JSON logs, security headers, Prometheus `/metrics`.
- **Storage:** SQLAlchemy - SQLite by default, PostgreSQL via `RETENTIONIQ_DATABASE_URL`. Uploads and results go through a blob-store interface (local filesystem; S3-ready).
- **Background jobs:** database-backed queue with optimistic claiming, retries with backoff, dead-lettering, and a scheduler.
- **Transactional outbox:** webhook events are written in the same transaction as the business change and delivered at-least-once.

## Model performance (held-out test set, n = 1,409)

| Metric | Value | 95% CI |
|---|---|---|
| ROC-AUC | 0.843 | 0.819 - 0.865 |
| PR-AUC (random baseline = 0.265) | 0.654 | 0.603 - 0.704 |
| Brier score | 0.137 | |
| Precision / recall at the cost-optimised threshold (0.29) | 0.529 / 0.775 | |
| Precision among the top-100 riskiest customers | 0.80 | |

Logistic regression, random forest and XGBoost are compared with stratified CV; all three are
within noise of each other. The subgroup check flags higher recall for senior citizens (0.90 vs
0.73) for review before real-world use. Details: [reports/evaluation_report.md](reports/evaluation_report.md)
and [reports/model_card.md](reports/model_card.md).

## Architecture

```
Streamlit UI ─┐                                   ┌─► webhooks (outbox) ─► CRM / sandbox
FastAPI ──────┼─► PredictionService ─► Repository ┤
Worker ───────┘   validate → score → explain →    └─► SQLite / PostgreSQL (org-scoped)
                  tier → policy → experiment arm
                         ▲
        make train ──────┘ versioned artifacts, model cards, registry (candidate/staging/production)
```

```
app.py, app_pages/, ui/    Streamlit app (sign-in, role-aware navigation, 14 pages)
api/                       FastAPI app, schemas, auth / RBAC / rate-limit dependencies
src/retentioniq/
  auth/                    Password hashing, roles and permissions
  data/  features/         Data contract and shared preprocessing
  models/                  Train (incl. tuning + feedback labels), evaluate, registry, explain
  decisioning/             Risk tiers, policy engine, action workflow
  experiments/             Registry, statistics, uplift modelling
  integrations/            Webhooks (signing, retries, SSRF guard), CRM exports
  jobs/                    Queue, handlers, worker + scheduler
  monitoring/              Drift, delayed-label performance, shadow comparison, alerts
  assistant/               Semantic layer for natural-language analytics
  storage/                 Engine, schema, tenant repository, accounts, blob store
configs/                   settings.yaml (thresholds, gates, schedules, limits), playbooks.yaml
scripts/                   seed_demo, make_samples, registry_cli
tests/                     126 tests: contract, auth, tenancy, API, jobs, webhooks, experiments, model, UI
```

Design decisions: [docs/architecture.md](docs/architecture.md) · alert runbooks: [docs/runbooks.md](docs/runbooks.md)

## API examples

```bash
curl -s localhost:8000/health
```

```bash
curl -s -X POST "localhost:8000/v1/predictions?persist=true" -H "Authorization: Bearer $RIQ_KEY" -H "Content-Type: application/json" -d '{"customerID":"DEMO-0001","gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":3,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":84.5,"TotalCharges":250.3}'
```

```bash
curl -s -X POST localhost:8000/v1/jobs/batch-scoring -H "Authorization: Bearer $RIQ_KEY" -F file=@data/samples/batch_clean.csv
```

Other endpoints: predictions (single, batch, lookup), customer retention summary, action
recommend / status / outcome, job status and results, webhooks and deliveries, experiments and
analysis, assistant query, model list / retrain / status, monitoring summary.

## Testing

```bash
make test
```

126 tests cover the data contract, authentication and lockout, API keys and RBAC, tenant
isolation (repository and API), the job queue and scheduler, webhook signing / retries /
dead-lettering / SSRF guard, experiment statistics and uplift recovery, the assistant's query
validation, model regression floors and lifecycle, the REST contract, and a render test for every page.

## Limitations

- Snapshot data without timestamps, so the split is stratified random rather than time-based.
- No real intervention data: action effectiveness values are assumptions until measured by an
  experiment; the demo's experiments correctly report "not enough outcomes" / no detectable effect.
- Refreshing the browser signs you out of the Streamlit app (session state is per connection).
- Docker, Compose and CI files describe the earlier single-service layout and are being updated
  for the worker and PostgreSQL.
