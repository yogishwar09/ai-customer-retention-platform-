
import streamlit as st

from retentioniq.jobs.queue import JobQueue
from retentioniq.jobs.worker import run_once
from retentioniq.storage.blobs import store
from ui import common

db = common.db()
queue = JobQueue()
common.header("Operations", "Background jobs, schedules and the worker. Scheduled scoring, imports, webhook delivery "
                            "and retraining all run here.")

depth = queue.depth()
jobs = queue.list(db.org_id, 300)
common.kpi_row([
    common.kpi("Queued", f"{depth.get('queued', 0):,}"),
    common.kpi("Running", f"{depth.get('running', 0):,}"),
    common.kpi("Succeeded", f"{depth.get('succeeded', 0):,}"),
    common.kpi("Dead", f"{depth.get('dead', 0):,}", "exhausted retries - inspect and retry"),
])

tab1, tab2, tab3 = st.tabs(["Jobs", "Schedules", "Batch jobs"])
with tab1:
    c1, c2 = st.columns([3, 1])
    types = c1.multiselect("Job types", sorted(jobs["job_type"].unique()) if not jobs.empty else [])
    if c2.button("Run worker once now", help="Processes due jobs in this session (the background worker also runs)."):
        st.toast(f"Processed {run_once()} job(s)")
        st.rerun()
    view = jobs[jobs["job_type"].isin(types)] if types else jobs
    st.dataframe(view, hide_index=True, use_container_width=True, height=380,
                 column_config={"progress": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.0f%%")})
    if not view.empty:
        pick = st.selectbox("Inspect job", view["job_id"].tolist())
        job = queue.get(pick)
        a, b = st.columns(2)
        a.json({"payload": job["payload"], "result": job["result"]})
        if job["error"]:
            b.code(job["error"][:3000])
        if job["status"] == "dead" and common.can("manage_integrations"):
            if b.button("Retry job"):
                queue.retry(pick)
                st.rerun()
        if job["org_id"] == db.org_id and job["result"] and job["result"].get("result_uri") and common.can("export"):
            b.download_button("Download results", store().get(job["result"]["result_uri"]), f"{pick}_scored.csv")

with tab2:
    sched = queue.schedules()
    if sched.empty:
        common.empty_state("No schedules registered yet", "They appear after the worker's first tick.")
    else:
        sched["every"] = sched["interval_seconds"].map(lambda s: f"{s // 3600} h" if s >= 3600 else
                                                       f"{s // 60} min" if s >= 60 else f"{s} s")
        st.dataframe(sched[["name", "every", "enabled", "last_enqueued_at"]], hide_index=True,
                     use_container_width=True, column_config={"enabled": st.column_config.CheckboxColumn()})
        if common.can("manage_integrations"):
            c1, c2 = st.columns([2, 1])
            name = c1.selectbox("Schedule", sched["name"].tolist())
            enabled = bool(sched.set_index("name").loc[name, "enabled"])
            if c2.button("Pause" if enabled else "Resume"):
                queue.set_schedule_enabled(name, not enabled)
                db.audit(common.actor(), "schedule.paused" if enabled else "schedule.resumed", "schedule", name)
                st.rerun()
            if st.button("Run scheduled rescore for my organisation now"):
                jid = queue.enqueue("scheduled_rescore", {}, org_id=db.org_id, created_by=common.actor(),
                                    max_attempts=1)
                st.success(f"Queued `{jid}`")
    st.caption("Intervals come from `scheduler:` in configs/settings.yaml. Run a dedicated worker with "
               "`python -m retentioniq.jobs.worker` (Docker Compose does); the app also runs one in-process "
               "unless RETENTIONIQ_INPROCESS_WORKER=0.")

with tab3:
    bj = db.batch_jobs()
    if bj.empty:
        common.empty_state("No batch jobs", "")
    else:
        st.dataframe(bj.drop(columns=["org_id"]), hide_index=True, use_container_width=True, height=380)
