# Architecture

RetentionIQ is a **modular monolith**: one deployable codebase with clean
internal boundaries, so pieces can be split into services later if scale or
ownership requires it.

```
             ┌────────────── Streamlit UI (app.py, app_pages/) ──────────────┐
             │ Overview · Queue · Customer · Score · Batch · Segments ·      │
             │ Model lab · Monitoring · Governance                           │
             └───────────────┬───────────────────────────────────────────────┘
                             │            ┌─ FastAPI (api/) ─ /v1/... ─┐
                             ▼            ▼                            │
                 retentioniq.service.PredictionService  ◄──────────────┘
      validate ─► preprocess+score ─► explain ─► risk tier ─► policy ─► persist
         │                 │               │                     │          │
   data/schema.py   models/registry   models/explain   decisioning/   storage/db
   (data contract)  (versioned, SHA)  (TreeSHAP/linear) policy.py     (SQLite)
                             ▲
            models/train.py ─┘  reproducible pipeline → artifacts/models/<version>/
                                                    → reports/model_card.md
            monitoring/  drift (PSI), delayed-label performance, impact by arm
```

## Decision records

**ADR-1 · Streamlit first, API alongside.** Streamlit keeps the analyst UI fast to
build; FastAPI exposes the same service for integrations. Both call
`PredictionService.score`, so behaviour cannot diverge between channels.

**ADR-2 · One persisted pipeline.** Preprocessing lives inside the scikit-learn
`Pipeline` that is saved with the model. The previous version re-implemented
feature handling in several files, which caused silent category mismatches
(e.g. `"Bank transfer"` vs `"Bank transfer (automatic)"`).

**ADR-3 · Explicit data contract.** `data/schema.py` defines columns, allowed
values, ranges and cross-field rules. Unknown categories are rejected with a
message rather than one-hot-encoded to all zeros.

**ADR-4 · Model selection by PR-AUC + cost-based threshold.** Churn is the
minority class (26.5%), so accuracy is misleading. The threshold minimises an
explicit FN:FP cost ratio on validation data; the test set is used once.

**ADR-5 · Artifact integrity.** Pickled models execute code on load, so the
registry verifies the SHA-256 recorded at training time before `joblib.load`.

**ADR-6 · SQLite now, PostgreSQL later.** Tables mirror the blueprint entities
(predictions, explanations, actions, outcomes, labels, audit, batch jobs).
Queries are parameterised and portable; swapping the driver is the migration.

**ADR-7 · Advisory, policy-driven actions with a control group.** Recommendations
come from versioned YAML rules limited to contract features, ranked by an
expected-value estimate whose effectiveness term is labelled as an assumption.
A deterministic 10% control group makes impact measurable instead of assumed.
