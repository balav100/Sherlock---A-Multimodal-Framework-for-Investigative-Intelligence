from datetime import datetime, timezone
import json
import logging
import os
import time

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.domain import processing_states as states
from backend.firebase import database_ref
from backend.routes.cases import check_case_access
from backend.security import get_current_user
from backend.services.hybrid_rag import (
    ingest_case,
    answer_case,
    analysis_record_from_result,
    EXTRACTION_PIPELINE_VERSION,
    CHUNKING_PIPELINE_VERSION,
)
from backend.services.hybrid_rag import _active_generation
from backend.services import jobs as job_store


router = APIRouter(
    prefix="/cases",
    tags=["Hybrid Multimodal RAG"],
)


class AskRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=2000,
    )

    top_k: int = Field(
        default=6,
        ge=1,
        le=12,
    )


def _active_evidence(case_id: str) -> dict:
    records = database_ref(f"cases/{case_id}/evidence").get() or {}
    if not isinstance(records, dict):
        return {}
    return {
        key: value for key, value in records.items()
        if isinstance(value, dict) and value.get("lifecycle_status") != "deleted"
    }


def ingest_and_index(case_id: str, user):
    """Synchronous ingest used by tests and background jobs.

    HTTP clients should call the routed ingest endpoint, which returns 202 and
    runs this function on a worker thread.
    """
    records = _active_evidence(case_id)
    if not records:
        raise HTTPException(
            status_code=400,
            detail="No active evidence uploaded for this case.",
        )

    case_ref = database_ref(f"cases/{case_id}")
    case_ref.update({"status": "indexing"})
    try:
        result = ingest_case(
            case_id,
            records,
        )

        # Persist analysis status and extraction metadata.
        for evidence_id, record in records.items():
            if not isinstance(record, dict):
                continue
            update = {
                "analysis_status": record.get("analysis_status", "failed"),
                "analysis": record.get("analysis"),
                "analysis_error": record.get("analysis_error"),
                "graph_status": record.get("graph_status", "unknown"),
            }
            database_ref(
                f"cases/{case_id}/evidence/{evidence_id}"
            ).update(update)

        database_ref(
            f"cases/{case_id}/analysis"
        ).update(
            analysis_record_from_result(
                result,
                ingested_evidence_ids=records.keys(),
                live_evidence_ids=_active_evidence(case_id).keys(),
            )
        )
        case_ref.update({
            "status": "analysis_completed" if result.get("published") else "analysis_partial"
        })

        return result

    
    except Exception as exc:
        # Save failure statuses even if ingestion raises an error.
        persistence_failed = False

        for evidence_id, record in records.items():
            if (
                isinstance(record, dict)
                and record.get("analysis_status") == "failed"
            ):
                try:
                    database_ref(
                        f"cases/{case_id}/evidence/{evidence_id}"
                    ).update({
                        "analysis_status": "failed",
                        "analysis_error": record.get("analysis_error"),
                    })
                except Exception:
                    persistence_failed = True

        logging.getLogger(__name__).exception(
            "Hybrid indexing failed for case_id=%s", case_id
        )
        try:
            case_ref.update({"status": "analysis_failed"})
        except Exception:
            logging.getLogger(__name__).exception(
                "Could not persist failed case status for case_id=%s", case_id
            )
        raise HTTPException(
            status_code=500,
            detail=("Indexing failed. " +
                    ("Some failure statuses could not be saved; check server logs." if persistence_failed else "Check server logs.")),
        ) from None


def _execute_ingest_job(case_id: str, job_id: str, user: dict) -> None:
    job_store.update_job(
        case_id, job_id,
        processing_state=states.VALIDATING,
        status="running",
        started_at=datetime.now(timezone.utc).isoformat(),
        message="Validating evidence records.",
        progress=0.05,
    )
    job_store.update_job(
        case_id, job_id,
        processing_state=states.EXTRACTING,
        message="Extracting and indexing evidence.",
        progress=0.15,
    )
    for evidence_id in (job_store.get_job(case_id, job_id) or {}).get("evidence_ids") or []:
        try:
            job_store.set_evidence_state(case_id, evidence_id, states.EXTRACTING, analysis_status="extracting")
        except Exception:
            logging.getLogger(__name__).exception("Could not mark evidence extracting: %s", evidence_id)
    job_store.update_job(case_id, job_id, processing_state=states.INDEXING, message="Publishing index generation.", progress=0.8)
    try:
        result = ingest_and_index(case_id, user)
    except HTTPException as exc:
        job_store.update_job(
            case_id, job_id,
            processing_state=states.FAILED,
            status="failed",
            error=str(exc.detail),
            message=str(exc.detail),
            finished_at=datetime.now(timezone.utc).isoformat(),
            progress=1.0,
        )
        raise
    published = bool(result.get("published"))
    failed = int(result.get("failed_evidence") or 0)
    if published:
        final_state = states.COMPLETED
        case_status = "analysis_completed"
        message = "Index published."
    elif result.get("preserved_previous_generation"):
        final_state = states.PARTIALLY_COMPLETED
        case_status = "analysis_partial"
        message = result.get("message") or "Previous index preserved."
    else:
        final_state = states.FAILED
        case_status = "analysis_failed"
        message = result.get("message") or "Index generation was not published."
    evidence_ids = (job_store.get_job(case_id, job_id) or {}).get("evidence_ids") or []
    for evidence_id in evidence_ids:
        record = database_ref(f"cases/{case_id}/evidence/{evidence_id}").get() or {}
        analysis_status = record.get("analysis_status") if isinstance(record, dict) else None
        try:
            if analysis_status == "indexed":
                job_store.set_evidence_state(case_id, evidence_id, states.COMPLETED, analysis_status="indexed")
            elif analysis_status == "failed":
                job_store.set_evidence_state(case_id, evidence_id, states.FAILED, analysis_status="failed")
        except Exception:
            logging.getLogger(__name__).exception("Could not finalize evidence state: %s", evidence_id)
    job_store.update_job(
        case_id, job_id,
        processing_state=final_state,
        status="completed" if final_state == states.COMPLETED else "failed" if final_state == states.FAILED else "partial",
        result=result,
        message=message,
        progress=1.0,
        finished_at=datetime.now(timezone.utc).isoformat(),
        error=None if published else message,
    )
    database_ref(f"cases/{case_id}").update({"status": case_status})
    if failed and published:
        pass


@router.post("/{case_id}/ingest-index")
def start_ingest_index(
    case_id: str,
    user=Depends(get_current_user),
    mode: str | None = Query(default=None, description="async (default) or sync"),
):
    """Queue case indexing. Default is non-blocking (HTTP 202 + job_id)."""
    check_case_access(case_id, user["uid"])
    records = _active_evidence(case_id)
    if not records:
        raise HTTPException(status_code=400, detail="No active evidence uploaded for this case.")
    requested = (mode or os.getenv("SHERLOCK_INGEST_MODE", "async")).strip().lower()
    if requested == "sync":
        return ingest_and_index(case_id, user)

    job = job_store.create_ingest_job(case_id, user["uid"], list(records.keys()))
    job_id = job["job_id"]
    if job.get("processing_state") not in states.TERMINAL_STATES:
        job_store.start_ingest_thread(
            case_id,
            job_id,
            lambda: _execute_ingest_job(case_id, job_id, user),
        )
    return JSONResponse(status_code=202, content=job)


@router.get("/{case_id}/jobs")
def list_case_jobs(case_id: str, user=Depends(get_current_user)):
    check_case_access(case_id, user["uid"])
    jobs = list(job_store.list_jobs(case_id).values())
    jobs = [item for item in jobs if isinstance(item, dict)]
    jobs.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return {"jobs": jobs, "active_job": job_store.active_ingest_job(case_id)}


@router.get("/{case_id}/jobs/{job_id}")
def get_case_job(case_id: str, job_id: str, user=Depends(get_current_user)):
    check_case_access(case_id, user["uid"])
    job = job_store.get_job(case_id, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@router.get("/{case_id}/jobs/{job_id}/events")
def stream_case_job(case_id: str, job_id: str, user=Depends(get_current_user)):
    """SSE snapshot stream of job state. Server state remains authoritative."""
    check_case_access(case_id, user["uid"])
    if not job_store.get_job(case_id, job_id):
        raise HTTPException(status_code=404, detail="Job not found.")

    def events():
        last = None
        deadline = time.time() + 3600
        while time.time() < deadline:
            job = job_store.get_job(case_id, job_id) or {}
            payload = json.dumps(job, default=str)
            if payload != last:
                yield f"event: job\ndata: {payload}\n\n"
                last = payload
            if job.get("processing_state") in states.TERMINAL_STATES:
                yield "event: done\ndata: {}\n\n"
                return
            time.sleep(1.0)
        yield "event: timeout\ndata: {}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@router.get("/{case_id}/index-status")
def get_index_status(case_id: str, user=Depends(get_current_user)):
    """Read-only status check for the saved FAISS index and chunk metadata."""
    check_case_access(case_id, user["uid"])

    try:
        folder, active_pointer = _active_generation(case_id)
    except FileNotFoundError:
        folder, active_pointer = None, None
    except (ValueError, OSError, json.JSONDecodeError):
        folder, active_pointer = None, None

    index_path = folder / "evidence.faiss" if folder else None
    metadata_path = folder / "chunks.json" if folder else None
    evidence_records = database_ref(f"cases/{case_id}/evidence").get() or {}
    evidence_records = {k: v for k, v in evidence_records.items()
                        if isinstance(v, dict) and v.get("lifecycle_status") != "deleted"}
    analysis_state = database_ref(f"cases/{case_id}/analysis").get() or {}

    per_evidence = {
        key: {
            "evidence_id": key,
            "filename": record.get("original_filename", key),
            "modality": record.get("modality", "unknown"),
            "chunks": 0,
            "status": record.get("analysis_status", "pending"),
            "processing_state": record.get("processing_state") or "UPLOADED",
        }
        for key, record in evidence_records.items()
    }
    active_job = job_store.active_ingest_job(case_id)
    base = {
        "case_id": case_id, "ready": False,
        "graph_status": analysis_state.get("graph_status", "unknown"),
        "total_chunks": 0,
        "total_evidence": len(evidence_records), "processed_evidence": 0,
        "evidence": list(per_evidence.values()),
        "reason": "No complete saved index was found for this case.",
        "active_job": active_job,
        "case_processing_state": (database_ref(f"cases/{case_id}").get() or {}).get("processing_state"),
    }
    if active_job and active_job.get("processing_state") not in states.TERMINAL_STATES:
        base["reason"] = (
            f"Indexing is in progress ({active_job.get('job_id')}: "
            f"{active_job.get('processing_state')})."
        )
        return base
    if analysis_state.get("index_status") == "stale":
        base["reason"] = "Evidence has changed since the last index. Rebuild the index before asking questions."
        return base
    if index_path is None or metadata_path is None or not index_path.is_file() or not metadata_path.is_file():
        return base
    if active_pointer is None:
        # Legacy flat layout predates the versioned generation manifest, so the
        # extraction pipeline that produced it cannot be verified. Require a
        # rebuild so answers never rely on an outdated extraction.
        base["reason"] = (
            "This index uses the legacy pre-generation layout. Rebuild the index "
            "before asking questions so answers use the current extraction pipeline."
        )
        base["rebuild_required"] = True
        return base

    try:
        if active_pointer is not None:
            manifest_path = folder / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("case_id") != case_id or manifest.get("schema_version") != 2:
                raise ValueError("Active index manifest is incompatible or belongs to another case.")
            if (
                manifest.get("extraction_pipeline_version") != EXTRACTION_PIPELINE_VERSION
                or manifest.get("chunking_pipeline_version") != CHUNKING_PIPELINE_VERSION
            ):
                base["reason"] = (
                    "The saved index was built with an older extraction/chunking "
                    "pipeline. Rebuild the index before asking questions."
                )
                base["rebuild_required"] = True
                return base
            base["generation"] = active_pointer.get("generation")
            base["evidence_manifest_sha256"] = manifest.get("evidence_manifest_sha256")
            current_manifest = []
            for evidence_id, record in evidence_records.items():
                sha = record.get("sha256")
                if not isinstance(sha, str) or not sha:
                    raise ValueError("A current evidence record has no SHA-256; index freshness cannot be verified.")
                current_manifest.append({
                    "evidence_id": str(evidence_id),
                    "sha256": sha.lower(),
                    "storage_path": str(record.get("storage_path") or ""),
                    "modality": str(record.get("modality", "unknown")),
                    "extraction_pipeline": manifest.get("extraction_pipeline_version"),
                })
            current_manifest.sort(key=lambda item: item["evidence_id"])
            import hashlib
            current_digest = hashlib.sha256(json.dumps(current_manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if current_digest != manifest.get("evidence_manifest_sha256"):
                base["reason"] = "The active index does not match the current evidence manifest. Re-index before asking questions."
                return base
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, list) or not metadata:
            raise ValueError("chunks.json is empty or has an invalid format.")
        for item in metadata:
            if not isinstance(item, dict):
                raise ValueError("chunks.json contains an invalid chunk record.")
            if item.get("case_id") != case_id:
                raise ValueError("Index metadata contains a different case ID.")
            for field in ("evidence_id", "chunk_id", "filename", "text"):
                if not isinstance(item.get(field), str) or not item[field].strip():
                    raise ValueError(f"An indexed chunk is missing {field}.")

        import faiss
        index = faiss.read_index(str(index_path))
        if index.ntotal != len(metadata):
            raise ValueError("FAISS vector count does not match chunks.json.")
        if index.ntotal <= 0:
            raise ValueError("The FAISS index contains no vectors.")

        counts, filenames, modalities = {}, {}, {}
        for item in metadata:
            eid = item["evidence_id"]
            counts[eid] = counts.get(eid, 0) + 1
            filenames[eid] = item["filename"]
            modalities[eid] = item.get("modality", "unknown")

        for eid, count in counts.items():
            if eid not in per_evidence:
                per_evidence[eid] = {
                    "evidence_id": eid, "filename": filenames[eid],
                    "modality": modalities[eid], "chunks": count,
                    "status": "indexed (metadata only)",
                }
            else:
                per_evidence[eid]["chunks"] = count
                per_evidence[eid]["status"] = "indexed"

        processed = sum(1 for item in per_evidence.values() if item["chunks"] > 0)
        return {
            "case_id": case_id, "ready": processed > 0,
            "total_chunks": len(metadata), "total_evidence": len(evidence_records),
            "processed_evidence": processed,
            "evidence": sorted(per_evidence.values(), key=lambda x: x["filename"].lower()),
            "index_path": str(folder),
            "generation": active_pointer.get("generation") if active_pointer else None,
            "reason": None if processed else "No evidence records are represented in the index.",
            "active_job": active_job,
        }
    except Exception as exc:
        base["reason"] = f"Saved index could not be validated: {type(exc).__name__}: {exc}"
        return base


@router.post("/{case_id}/ask")
def ask_case(
    case_id: str,
    payload: AskRequest,
    user=Depends(get_current_user),
):
    """Ask an evidence-grounded investigation question."""

    check_case_access(
        case_id,
        user["uid"],
    )
    analysis_state = database_ref(f"cases/{case_id}/analysis").get() or {}
    if analysis_state.get("index_status") == "stale":
        raise HTTPException(
            status_code=409,
            detail="Evidence has changed since the last index. Rebuild the index before asking questions.",
        )

    try:
        result = answer_case(
            case_id=case_id,
            question=payload.question,
            top_k=payload.top_k,
        )

        database_ref(
            f"cases/{case_id}/analysis/last_answer"
        ).set(result)

        return result

    
    except FileNotFoundError:
        logging.getLogger(__name__).exception(
            "Evidence file unavailable during investigation query"
        )
        raise HTTPException(
            status_code=409,
            detail="Required evidence is unavailable. Check the case evidence.",
        ) from None

    except ValueError:
        logging.getLogger(__name__).exception(
            "Invalid investigation query input or evidence state"
        )
        raise HTTPException(
            status_code=400,
            detail="The investigation query could not be processed.",
        ) from None

    except Exception:
        logging.getLogger(__name__).exception(
            "Investigation query failed"
        )
        raise HTTPException(
            status_code=500,
            detail="Investigation query failed. Check server logs for details.",
        ) from None
