"""Persisted ingest jobs. Firebase RTDB is the current store; PostgreSQL later."""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from backend.domain import processing_states as states
from backend.firebase import database_ref

logger = logging.getLogger(__name__)

JOB_INGEST = "ingest_index"
_JOB_THREADS: dict[str, threading.Thread] = {}
_JOB_GUARD = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_path(case_id: str, job_id: str) -> str:
    return f"cases/{case_id}/jobs/{job_id}"


def list_jobs(case_id: str) -> dict[str, dict]:
    records = database_ref(f"cases/{case_id}/jobs").get() or {}
    return records if isinstance(records, dict) else {}


def get_job(case_id: str, job_id: str) -> dict | None:
    if not job_id or any(char in job_id for char in ".#$[]/"):
        return None
    record = database_ref(_job_path(case_id, job_id)).get()
    return record if isinstance(record, dict) else None


def active_ingest_job(case_id: str) -> dict | None:
    case = database_ref(f"cases/{case_id}").get() or {}
    job_id = case.get("active_job_id") if isinstance(case, dict) else None
    if isinstance(job_id, str) and job_id:
        job = get_job(case_id, job_id)
        if job and job.get("processing_state") not in states.TERMINAL_STATES:
            return job
    for job in list_jobs(case_id).values():
        if (
            isinstance(job, dict)
            and job.get("job_type") == JOB_INGEST
            and job.get("processing_state") not in states.TERMINAL_STATES
        ):
            return job
    return None


def _worker_is_alive(job_id: str) -> bool:
    """True only while this process still has a running worker for the job."""
    thread = _JOB_THREADS.get(job_id)
    return thread is not None and thread.is_alive()


def create_ingest_job(case_id: str, owner_id: str, evidence_ids: list[str]) -> dict:
    existing = active_ingest_job(case_id)
    if existing:
        job_id = existing.get("job_id")
        state = existing.get("processing_state")
        # A job left in an in-flight state by a worker that is no longer running was
        # interrupted (process restart, crash, or a worker that died before writing a
        # terminal state). Its persisted state cannot be resumed directly, so put it
        # back in the queue and let the pipeline validate from the start.
        if (
            isinstance(job_id, str)
            and job_id
            and state not in states.TERMINAL_STATES
            and state != states.QUEUED
            and not _worker_is_alive(job_id)
        ):
            logger.warning(
                "Re-queuing interrupted ingest job: case_id=%s job_id=%s state=%s",
                case_id,
                job_id,
                state,
            )
            existing = update_job(
                case_id,
                job_id,
                processing_state=states.QUEUED,
                status="queued",
                message="Re-queued after an interrupted run.",
                progress=0.0,
                error=None,
                started_at=None,
                finished_at=None,
            )
        return existing
    job_id = "JOB-" + uuid4().hex[:12].upper()
    payload = {
        "job_id": job_id,
        "case_id": case_id,
        "owner_id": owner_id,
        "job_type": JOB_INGEST,
        "processing_state": states.QUEUED,
        "status": "queued",
        "progress": 0.0,
        "message": "Ingest queued.",
        "evidence_ids": evidence_ids,
        "result": None,
        "error": None,
        "created_at": _now(),
        "updated_at": _now(),
        "started_at": None,
        "finished_at": None,
    }
    database_ref(_job_path(case_id, job_id)).set(payload)
    database_ref(f"cases/{case_id}").update({
        "status": "indexing",
        "active_job_id": job_id,
        "processing_state": states.QUEUED,
    })
    for evidence_id in evidence_ids:
        database_ref(f"cases/{case_id}/evidence/{evidence_id}").update({
            "processing_state": states.QUEUED,
            "analysis_status": "queued",
        })
    return payload


def update_job(case_id: str, job_id: str, **fields: Any) -> dict:
    current = get_job(case_id, job_id) or {}
    if fields.get("processing_state"):
        fields["processing_state"] = states.transition(
            current.get("processing_state"),
            fields["processing_state"],
        )
    fields["updated_at"] = _now()
    database_ref(_job_path(case_id, job_id)).update(fields)
    merged = {**current, **fields}
    case_fields = {}
    if "processing_state" in fields:
        case_fields["processing_state"] = fields["processing_state"]
    if fields.get("processing_state") in states.TERMINAL_STATES:
        case_fields["active_job_id"] = job_id
    if case_fields:
        database_ref(f"cases/{case_id}").update(case_fields)
    return merged


def set_evidence_state(case_id: str, evidence_id: str, state: str, **extra: Any) -> None:
    record = database_ref(f"cases/{case_id}/evidence/{evidence_id}").get() or {}
    current = record.get("processing_state") if isinstance(record, dict) else None
    payload = {"processing_state": states.transition(current or states.UPLOADED, state), **extra}
    database_ref(f"cases/{case_id}/evidence/{evidence_id}").update(payload)


def start_ingest_thread(case_id: str, job_id: str, worker: Callable[[], None]) -> None:
    """Run ingest off the request thread so GET status remains servable."""
    with _JOB_GUARD:
        existing = _JOB_THREADS.get(job_id)
        if existing is not None and existing.is_alive():
            return

        def _run() -> None:
            try:
                worker()
            except Exception:
                logger.exception("Ingest job crashed: case_id=%s job_id=%s", case_id, job_id)
                try:
                    update_job(
                        case_id,
                        job_id,
                        processing_state=states.FAILED,
                        status="failed",
                        error="Indexing failed. Check server logs.",
                        finished_at=_now(),
                        progress=1.0,
                    )
                    database_ref(f"cases/{case_id}").update({"status": "analysis_failed"})
                except Exception:
                    logger.exception("Could not persist crashed job: job_id=%s", job_id)

        thread = threading.Thread(target=_run, name=f"sherlock-ingest-{job_id}", daemon=True)
        _JOB_THREADS[job_id] = thread
        thread.start()
