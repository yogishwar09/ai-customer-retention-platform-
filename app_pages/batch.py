import io

import pandas as pd
import streamlit as st

from retentioniq import config
from retentioniq.data.schema import SchemaError, data_dictionary, validate_frame
from ui import common

svc = common.service()
db = common.db()
common.header("Batch scoring",
              "Upload → map columns → validate → score → download. Bad rows are reported, never silently scored.")

limits = config.settings()["uploads"]
samples = {
    "Clean synthetic file (300 rows)": "batch_clean.csv",
    "File with deliberate errors (60 rows)": "batch_with_errors.csv",
    "Drifted population (400 rows)": "batch_drifted.csv",
}

st.markdown("**1 · Choose a file**")
c1, c2 = st.columns([1.4, 1])
upload = c1.file_uploader("CSV with one customer per row", type=["csv"],
                          help=f"Max {limits['max_megabytes']} MB / {limits['max_rows']:,} rows. "
                               "Columns are matched case-insensitively.")
sample = c2.selectbox("…or use a sample file", ["—", *samples])
template = (config.SAMPLES_DIR / "template.csv")
if template.exists():
    c2.download_button("Download CSV template", template.read_bytes(), "retentioniq_template.csv", "text/csv")

with st.expander("Data dictionary (required columns)"):
    st.dataframe(data_dictionary(), hide_index=True, use_container_width=True)

raw_bytes, filename = None, None
if upload is not None:
    raw_bytes, filename = upload.getvalue(), upload.name
elif sample != "—":
    path = config.SAMPLES_DIR / samples[sample]
    if path.exists():
        raw_bytes, filename = path.read_bytes(), path.name
if raw_bytes is None:
    st.stop()

if len(raw_bytes) > limits["max_megabytes"] * 1024 * 1024:
    st.error(f"File is larger than {limits['max_megabytes']} MB.")
    st.stop()
try:
    df = pd.read_csv(io.BytesIO(raw_bytes), dtype=str, keep_default_na=False, na_values=[""],
                     encoding="utf-8-sig")
except UnicodeDecodeError:
    st.error("File is not UTF-8 encoded. Re-save it as UTF-8 CSV.")
    st.stop()
except Exception as exc:  # malformed CSV
    st.error(f"Could not parse the CSV: {exc}")
    st.stop()
if len(df) == 0:
    st.error("The file has a header but no rows.")
    st.stop()
if len(df) > limits["max_rows"]:
    st.error(f"{len(df):,} rows exceeds the {limits['max_rows']:,} row limit. Split the file.")
    st.stop()

st.markdown("**2 · Preview & column mapping**")
st.dataframe(df.head(5), use_container_width=True, hide_index=True)
try:
    result = validate_frame(df)
except SchemaError as exc:
    st.error(str(exc))
    st.stop()

s = result.summary
m1, m2, m3 = st.columns(3)
m1.markdown("**Renamed columns**<br>" + (", ".join(f"`{k}` → `{v}`" for k, v in s["renamed_columns"].items())
                                         or "<span class='muted'>none</span>"), unsafe_allow_html=True)
m2.markdown("**Ignored columns**<br>" + (", ".join(f"`{c}`" for c in s["ignored_columns"])
                                         or "<span class='muted'>none</span>"), unsafe_allow_html=True)
m3.markdown("**Customer IDs**<br>" + ("present" if result.has_ids else
                                      "<b style='color:#B7791F'>◆ missing - results can't be stored "
                                      "per customer</b>"), unsafe_allow_html=True)

st.markdown("**3 · Validation report**")
common.kpi_row([
    common.kpi("Rows in file", f"{s['rows_total']:,}"),
    common.kpi("Accepted", f"{s['rows_accepted']:,}", f"{s['rows_accepted'] / s['rows_total']:.0%}"),
    common.kpi("Rejected", f"{s['rows_rejected']:,}", "see row-level errors"),
    common.kpi("Accepted with warnings", f"{s['rows_with_warnings']:,}"),
])
if not result.rejected.empty:
    with st.expander(f"Rejected rows ({len(result.rejected)})", expanded=True):
        st.dataframe(result.rejected[["row_number", "errors"]], hide_index=True, use_container_width=True)
        st.download_button("Download error report (CSV)", result.rejected.to_csv(index=False).encode(),
                           f"{filename.rsplit('.', 1)[0]}_errors.csv", "text/csv")
warn_rows = [(i, w) for i, w in result.row_warnings.items() if w]
if warn_rows:
    with st.expander(f"Warnings ({len(warn_rows)})"):
        st.dataframe(pd.DataFrame([{"accepted_row": i + 1, "warnings": "; ".join(w)} for i, w in warn_rows]),
                     hide_index=True, use_container_width=True)
if s["rows_accepted"] == 0:
    st.error("No valid rows to score.")
    st.stop()

st.markdown("**4 · Score**")
persist_allowed = result.has_ids and common.can("agent")
o1, o2 = st.columns(2)
persist = o1.toggle("Store results (adds customers to the queue and monitoring)", value=persist_allowed,
                    disabled=not persist_allowed,
                    help="Requires customer IDs and the agent role.")
background = o2.toggle("Run as background job", value=s["rows_total"] > 2000,
                       help="Queues the file for the worker so large uploads never block the page. "
                            "Track it in Operations.")
if background and st.button(f"Queue {s['rows_total']:,} rows for background scoring", type="primary"):
    import uuid

    from retentioniq.jobs.queue import JobQueue
    from retentioniq.storage.blobs import store
    job_id = f"job_{uuid.uuid4().hex[:16]}"
    uri = store().put(f"{db.org_id}/uploads/{job_id}.csv", raw_bytes)
    JobQueue().enqueue("batch_score", {"upload_uri": uri, "filename": filename, "persist": persist,
                                       "source": "upload_job", "actor": common.actor(), "batch_job_id": job_id},
                       org_id=db.org_id, created_by=common.actor(), job_id=job_id)
    db.audit(common.actor(), "job.submitted", "job", job_id, {"type": "batch_score", "file": filename})
    st.session_state["queued_job"] = job_id

if job_id := st.session_state.get("queued_job"):
    from retentioniq.jobs.queue import JobQueue
    from retentioniq.storage.blobs import store
    job = JobQueue().get(job_id)
    if job:
        st.info(f"Background job `{job_id}` · status **{job['status']}** · progress {(job['progress'] or 0):.0%}")
        c1, c2, c3 = st.columns([1, 1.4, 3])
        c1.button("Refresh status")
        if job["status"] == "succeeded" and job["result"].get("result_uri"):
            c2.download_button("Download scored results", store().get(job["result"]["result_uri"]),
                               f"{job_id}_scored.csv", "text/csv", type="primary")
        if job["status"] in ("dead", "failed") or (job["result"] or {}).get("status") == "failed":
            st.error(job.get("error") or job["result"].get("error"))
    st.stop()

if not background and st.button(f"Score {s['rows_accepted']:,} valid rows", type="primary"):
    job_id = db.start_batch_job(filename, common.actor(), svc.model.version) if persist else None
    bar = st.progress(0.0, text="Scoring…")
    try:
        batch = svc.score(df, source="batch", repo=db, persist=persist, batch_job_id=job_id,
                          progress=lambda f: bar.progress(f, text=f"Scoring… {f:.0%}"), chunk_size=250)
        if job_id:
            db.finish_batch_job(job_id, status="completed", summary=batch.validation.summary,
                                actor=common.actor())
        st.session_state["batch_result"] = (filename, batch.to_frame(), batch.latency_ms, job_id)
    except Exception as exc:
        if job_id:
            db.finish_batch_job(job_id, status="failed", error=str(exc)[:500], actor=common.actor())
        st.error(f"Scoring failed: {exc}")
        st.stop()
    bar.empty()

res = st.session_state.get("batch_result")
if res and res[0] == filename:
    _, out, latency, job_id = res
    st.success(f"Scored {len(out):,} customers in {latency / 1000:.1f}s"
               + (f" · stored as batch job `{job_id}`" if job_id else " · not stored"))
    tiers = out["risk_tier"].value_counts()
    common.kpi_row([
        common.kpi("High risk", f"{tiers.get('high', 0):,}"),
        common.kpi("Medium risk", f"{tiers.get('medium', 0):,}"),
        common.kpi("Low risk", f"{tiers.get('low', 0):,}"),
        common.kpi("Revenue at risk (est.)", common.money(out["revenue_at_risk_estimate"].sum())),
    ])
    st.dataframe(out[["customerID", "churn_probability", "risk_tier", "reason_codes", "recommended_action",
                      "data_quality_status", "model_version", "scored_at"]]
                 .sort_values("churn_probability", ascending=False),
                 hide_index=True, use_container_width=True, height=360,
                 column_config={"churn_probability": st.column_config.ProgressColumn(
                     "Est. churn risk", format="%.2f", min_value=0, max_value=1)})
    st.download_button("Download scored results (CSV)", out.to_csv(index=False).encode(),
                       f"{filename.rsplit('.', 1)[0]}_scored.csv", "text/csv", type="primary")
    if len(out) >= config.settings()["monitoring"]["min_rows_for_drift"]:
        with st.expander("Drift of this file vs. training data"):
            from retentioniq.data.schema import FEATURE_COLUMNS
            drift = svc.drift(out[FEATURE_COLUMNS], out["churn_probability"].to_numpy())
            st.dataframe(drift, hide_index=True, use_container_width=True,
                         column_config={"psi": st.column_config.NumberColumn("PSI", format="%.3f")})
