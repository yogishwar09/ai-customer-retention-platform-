from retentioniq import config
from retentioniq.jobs.queue import JobQueue
from retentioniq.jobs.worker import run_job, run_once
from retentioniq.storage.blobs import LocalBlobStore, store


def test_claim_complete_and_isolated_claims(trained):
    q = JobQueue()
    jid = q.enqueue("dispatch_webhooks", {})
    job = q.claim("w1")
    assert job["job_id"] == jid and job["status"] == "running"
    assert q.claim("w2", job_types=["dispatch_webhooks"]) is None  # already claimed
    q.complete(jid, {"ok": True})
    assert q.get(jid)["status"] == "succeeded"


def test_failures_retry_then_dead(trained):
    q = JobQueue()
    jid = q.enqueue("no_such_handler", {}, max_attempts=2)
    job = q.claim(job_types=["no_such_handler"])
    assert q.fail(jid, "boom") == "queued"
    assert q.get(jid)["run_after"] > job["run_after"]  # exponential backoff
    q.engine.dispose()
    from sqlalchemy import text
    with q.engine.begin() as c:
        c.execute(text("UPDATE jobs SET run_after = '2000-01-01T00:00:00+00:00' WHERE job_id = :i"), dict(i=jid))
    job = q.claim(job_types=["no_such_handler"])
    assert run_job(q, job) == "dead"
    q.retry(jid)
    assert q.get(jid)["status"] == "queued"


def test_schedule_tick_is_idempotent(trained):
    q = JobQueue()
    first = q.schedule_tick({"fast_task": 10, "slow_task": 3600})
    assert first == ["fast_task"]  # long intervals first fire one interval after registration
    assert q.schedule_tick({"fast_task": 10, "slow_task": 3600}) == []


def test_async_batch_scoring_job(service, org):
    data = (config.SAMPLES_DIR / "batch_with_errors.csv").read_bytes()
    uri = store().put(f"{org.org_id}/uploads/test.csv", data)
    q = JobQueue()
    jid = q.enqueue("batch_score", {"upload_uri": uri, "filename": "test.csv", "persist": True, "actor": "t"},
                    org_id=org.org_id)
    while (job := q.claim(job_types=["batch_score"])) and job["job_id"] != jid:
        q.complete(job["job_id"])
    assert run_job(q, job) == "succeeded"
    res = q.get(jid)["result"]
    assert res["rows_accepted"] == 53 and res["rows_rejected"] == 7
    assert b"errors" in store().get(res["error_report_uri"])
    assert org.get_batch_job(jid)["status"] == "completed"
    assert len(org.latest_scores()) == 53


def test_inbox_import_picks_up_files(service, org):
    from retentioniq.storage.accounts import Accounts
    slug = Accounts().get_org(org.org_id)["slug"]
    folder = config.PROJECT_ROOT / config.settings()["jobs"]["inbox_dir"] / slug
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "drop.csv").write_bytes((config.SAMPLES_DIR / "template.csv").read_bytes())
    try:
        q = JobQueue()
        q.enqueue("inbox_import", {})
        run_once(q, schedule=False)
        assert not (folder / "drop.csv").exists()
        assert len(org.latest_scores()) == 3
    finally:
        import shutil
        shutil.rmtree(folder, ignore_errors=True)


def test_blob_store_blocks_traversal(tmp_path):
    import pytest
    s = LocalBlobStore(tmp_path)
    with pytest.raises(ValueError):
        s.put("org/../../evil.txt", b"x")
    with pytest.raises(ValueError):
        s.get("local://../../etc/passwd")
    uri = s.put("org/ok file;rm.csv", b"x")
    assert s.get(uri) == b"x" and ";" not in uri
