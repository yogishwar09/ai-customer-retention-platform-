"""Background worker: runs the scheduler tick and executes queued jobs.

    python -m retentioniq.jobs.worker            # long-running process
    python -m retentioniq.jobs.worker --once     # drain the queue once (cron/CI)

The Streamlit app can also run one in a daemon thread
(RETENTIONIQ_INPROCESS_WORKER=1, the default for single-host demos).
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
import traceback
from typing import Any

from retentioniq import config
from retentioniq.jobs.handlers import HANDLERS, JobContext
from retentioniq.jobs.queue import WORKER_ID, JobQueue

log = logging.getLogger("retentioniq.worker")

_service_cache: dict[str, Any] = {}


def _service():
    from retentioniq.models import registry
    from retentioniq.service import PredictionService
    key = f"{registry.production_version()}|{registry.staging_version()}"
    if key not in _service_cache:
        _service_cache.clear()
        _service_cache[key] = PredictionService()
    return _service_cache[key]


def run_job(queue: JobQueue, job: dict) -> str:
    handler = HANDLERS.get(job["job_type"])
    if handler is None:
        queue.fail(job["job_id"], f"No handler for {job['job_type']}")
        return "dead"
    try:
        result = handler(job, JobContext(queue, _service))
        queue.complete(job["job_id"], result)
        return "succeeded"
    except Exception as exc:
        status = queue.fail(job["job_id"], f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}")
        log.warning("job %s (%s) failed: %s -> %s", job["job_id"], job["job_type"], exc, status)
        return status


def run_once(queue: JobQueue | None = None, *, schedule: bool = True, max_jobs: int = 100) -> int:
    queue = queue or JobQueue()
    if schedule:
        queue.schedule_tick(config.settings().get("scheduler", {}))
    n = 0
    while n < max_jobs and (job := queue.claim()):
        run_job(queue, job)
        n += 1
    return n


def loop(poll_seconds: float = 2.0, stop: threading.Event | None = None) -> None:
    queue = JobQueue()
    log.info("worker %s started", WORKER_ID)
    while not (stop and stop.is_set()):
        try:
            if run_once(queue) == 0:
                time.sleep(poll_seconds)
        except Exception:  # keep the worker alive; errors are recorded per job
            log.exception("worker tick failed")
            time.sleep(poll_seconds * 5)


_thread: threading.Thread | None = None


def start_background_thread() -> threading.Thread:
    global _thread
    if _thread is None or not _thread.is_alive():
        _thread = threading.Thread(target=loop, name="retentioniq-worker", daemon=True)
        _thread.start()
    return _thread


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.once:
        print(f"processed {run_once()} job(s)")
    else:
        loop()
