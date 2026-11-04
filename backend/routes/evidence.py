"""Evidence upload, listing and lifecycle endpoints."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from backend.config import MAX_UPLOAD_MB
from backend.firebase import database_ref
from backend.routes.cases import check_case_access
from backend.security import get_current_user
from backend.services.hybrid_rag import analysis_record_from_result, ingest_case
from backend.services.evidence_storage import (
    STORAGE_ROOT, delete_object, safe_object_path, store_upload,
)

router = APIRouter(prefix="/cases", tags=["Evidence"])
logger = logging.getLogger(__name__)
EVIDENCE_ROOT = STORAGE_ROOT  # compatibility alias; configured persistent root

ALLOWED_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".jpg", ".jpeg", ".png", ".webp",
    ".pdf", ".txt", ".csv", ".json", ".wav", ".mp3",
}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
AUDIO_EXTENSIONS = {".wav", ".mp3"}
TABULAR_EXTENSIONS = {".csv", ".json"}


def get_modality(extension: str) -> str:
    if extension in VIDEO_EXTENSIONS:
        return "video"
    if extension in IMAGE_EXTENSIONS:
        return "image"
    if extension in AUDIO_EXTENSIONS:
        return "audio"
    if extension in TABULAR_EXTENSIONS:
        return "tabular"
    return "document"


def _mark_index_stale(case_id: str, reason: str) -> None:
    """Invalidate the logical index status; index rebuild is handled by indexing phase."""
    database_ref(f"cases/{case_id}/analysis").update({
        "index_status": "stale",
        "stale_reason": reason,
        "stale_at": datetime.now(timezone.utc).isoformat(),
    })
    database_ref(f"cases/{case_id}").update({"status": "evidence_changed"})


def _active_evidence(case_id: str) -> dict:
    """Evidence records belonging to the case that have not been deleted."""
    records = database_ref(f"cases/{case_id}/evidence").get() or {}
    if not isinstance(records, dict):
        return {}
    return {
        key: value for key, value in records.items()
        if isinstance(value, dict) and value.get("lifecycle_status") != "deleted"
    }


def _refresh_evidence_count(case_id: str) -> int:
    count = len(_active_evidence(case_id))
    database_ref(f"cases/{case_id}").update({"evidence_count": count})
    return count


@router.post("/{case_id}/evidence", status_code=201)
def upload_evidence(case_id: str, file: UploadFile = File(...), user=Depends(get_current_user)):
    check_case_access(case_id, user["uid"])
    original_name = Path(file.filename or "").name.strip()
    if not original_name or original_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="A filename is required.")
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {extension or 'unknown'}")

    evidence_id = "EVD-" + uuid4().hex[:12].upper()
    storage_path = f"cases/{case_id}/evidence/{evidence_id}/original"
    try:
        stored = store_upload(
            file.file, storage_path, extension, MAX_UPLOAD_MB * 1024 * 1024, root=EVIDENCE_ROOT,
        )
    except OverflowError:
        raise HTTPException(status_code=413, detail=f"File exceeds the {MAX_UPLOAD_MB} MB limit.") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except OSError:
        logger.exception("Evidence storage failed: case_id=%s evidence_id=%s", case_id, evidence_id)
        raise HTTPException(status_code=500, detail="Evidence could not be stored. Check server logs.") from None

    now = datetime.now(timezone.utc).isoformat()
    existing_records = database_ref(f"cases/{case_id}/evidence").get() or {}
    duplicate_of = next((
        key for key, value in existing_records.items()
        if isinstance(value, dict)
        and value.get("lifecycle_status") != "deleted"
        and value.get("sha256") == stored["sha256"]
    ), None) if isinstance(existing_records, dict) else None
    evidence_data = {
        "evidence_id": evidence_id,
        "case_id": case_id,
        "original_filename": original_name,
        "storage_path": storage_path,
        "storage_backend": "local",
        "sha256": stored["sha256"],
        "size_bytes": stored["size_bytes"],
        "duplicate_content_of": duplicate_of,
        "modality": get_modality(extension),
        "content_type": stored["detected_content_type"],
        "declared_content_type": file.content_type or "application/octet-stream",
        "file_extension": extension,
        "uploaded_by": user["uid"],
        "uploaded_at": now,
        "lifecycle_status": "active",
        "analysis_status": "pending",
        "processing_state": "UPLOADED",
    }
    evidence_ref = database_ref(f"cases/{case_id}/evidence/{evidence_id}")
    try:
        evidence_ref.set(evidence_data)
    except Exception:
        logger.exception("Evidence metadata persistence failed: case_id=%s evidence_id=%s", case_id, evidence_id)
        try:
            delete_object(storage_path, root=EVIDENCE_ROOT)
        except OSError:
            logger.exception("Evidence rollback failed: evidence_id=%s", evidence_id)
        raise HTTPException(status_code=503, detail="Evidence metadata could not be saved. Please retry.") from None

    # Count is a derived/reconciled field. A count update failure must not delete
    # a file whose authoritative evidence record has already been committed.
    try:
        _refresh_evidence_count(case_id)
        database_ref(f"cases/{case_id}").update({"status": "evidence_uploaded"})
    except Exception:
        logger.exception("Evidence count reconciliation required: case_id=%s", case_id)
        evidence_data["count_reconciliation_required"] = True
        try:
            evidence_ref.update({"count_reconciliation_required": True})
        except Exception:
            logger.exception("Could not mark evidence count reconciliation: evidence_id=%s", evidence_id)

    try:
        _mark_index_stale(case_id, "new_evidence_uploaded")
    except Exception:
        logger.exception("Could not mark index stale after upload: case_id=%s", case_id)
    return evidence_data


@router.get("/{case_id}/evidence")
def list_evidence(case_id: str, user=Depends(get_current_user)):
    check_case_access(case_id, user["uid"])
    evidence = database_ref(f"cases/{case_id}/evidence").get() or {}
    items = []
    if isinstance(evidence, dict):
        for key, record in evidence.items():
            if isinstance(record, dict) and record.get("lifecycle_status") != "deleted":
                items.append({**record, "evidence_id": record.get("evidence_id") or key})
    items.sort(key=lambda item: item.get("uploaded_at", ""))
    return {"evidence": items, "count": len(items)}


@router.put("/{case_id}/evidence/{evidence_id}")
def replace_evidence(case_id: str, evidence_id: str, file: UploadFile = File(...), user=Depends(get_current_user)):
    """Replace an exhibit without destroying the previous binary before commit."""
    check_case_access(case_id, user["uid"])
    if not evidence_id or any(char in evidence_id for char in ".#$[]/"):
        raise HTTPException(status_code=404, detail="Evidence not found.")
    ref = database_ref(f"cases/{case_id}/evidence/{evidence_id}")
    old_record = ref.get()
    if not isinstance(old_record, dict) or old_record.get("lifecycle_status") == "deleted":
        raise HTTPException(status_code=404, detail="Evidence not found.")
    original_name = Path(file.filename or "").name.strip()
    extension = Path(original_name).suffix.lower()
    if not original_name or extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="A supported filename and extension are required.")
    version_id = uuid4().hex[:12]
    new_storage_path = f"cases/{case_id}/evidence/{evidence_id}/versions/{version_id}/original"
    try:
        stored = store_upload(file.file, new_storage_path, extension, MAX_UPLOAD_MB * 1024 * 1024, root=EVIDENCE_ROOT)
    except OverflowError:
        raise HTTPException(status_code=413, detail=f"File exceeds the {MAX_UPLOAD_MB} MB limit.") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except OSError:
        logger.exception("Replacement binary storage failed: evidence_id=%s", evidence_id)
        raise HTTPException(status_code=500, detail="Replacement file could not be stored.") from None

    previous_path = old_record.get("storage_path")
    updated = {
        **old_record,
        "original_filename": original_name,
        "storage_path": new_storage_path,
        "sha256": stored["sha256"],
        "size_bytes": stored["size_bytes"],
        "file_extension": extension,
        "modality": get_modality(extension),
        "content_type": stored["detected_content_type"],
        "declared_content_type": file.content_type or "application/octet-stream",
        "uploaded_by": user["uid"],
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "lifecycle_status": "active",
        "analysis_status": "pending",
        "processing_state": "UPLOADED",
        "analysis": None,
        "analysis_error": None,
        "replaced_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        ref.set(updated)
    except Exception:
        logger.exception("Replacement metadata commit failed: evidence_id=%s", evidence_id)
        try:
            delete_object(new_storage_path, root=EVIDENCE_ROOT)
        except OSError:
            logger.exception("Replacement rollback failed: evidence_id=%s", evidence_id)
        raise HTTPException(status_code=503, detail="Replacement metadata could not be saved; the previous evidence remains active.") from None

    try:
        from backend.services.graph_rag import delete_evidence_graph
        delete_evidence_graph(case_id, evidence_id)
    except Exception as exc:
        logger.exception("Graph cleanup required after evidence replacement: evidence_id=%s error=%s", evidence_id, type(exc).__name__)
    if previous_path and previous_path != new_storage_path:
        try:
            delete_object(previous_path, root=EVIDENCE_ROOT)
        except (OSError, ValueError):
            logger.exception("Old evidence version cleanup required: evidence_id=%s", evidence_id)
            ref.update({"cleanup_required_path": previous_path})
    try:
        _mark_index_stale(case_id, "evidence_replaced")
    except Exception:
        logger.exception("Could not mark index stale after replacement: case_id=%s", case_id)
    return updated


@router.delete("/{case_id}/evidence/{evidence_id}")
def delete_evidence(case_id: str, evidence_id: str, user=Depends(get_current_user)):
    check_case_access(case_id, user["uid"])
    if not evidence_id or any(char in evidence_id for char in ".#$[]/"):
        raise HTTPException(status_code=404, detail="Evidence not found.")
    ref = database_ref(f"cases/{case_id}/evidence/{evidence_id}")
    record = ref.get()
    if not isinstance(record, dict) or record.get("lifecycle_status") == "deleted":
        raise HTTPException(status_code=404, detail="Evidence not found.")
    storage_path = record.get("storage_path")
    if record.get("storage_backend", "local") != "local":
        raise HTTPException(status_code=501, detail="This storage backend does not support deletion yet.")
    # Persist an intermediate state so interrupted deletes can be reconciled.
    try:
        ref.update({"lifecycle_status": "deleting", "delete_started_at": datetime.now(timezone.utc).isoformat()})
        if storage_path:
            delete_object(storage_path, root=EVIDENCE_ROOT)
    except (OSError, ValueError):
        logger.exception("Evidence binary deletion failed: case_id=%s evidence_id=%s", case_id, evidence_id)
        try:
            ref.update({"lifecycle_status": "active"})
        except Exception:
            logger.exception("Could not restore evidence lifecycle after failed delete: evidence_id=%s", evidence_id)
        raise HTTPException(status_code=503, detail="Evidence file could not be deleted. Metadata was preserved for retry.") from None
    try:
        from backend.services.graph_rag import delete_evidence_graph
        delete_evidence_graph(case_id, evidence_id)
    except Exception as exc:
        logger.exception("Graph cleanup required after evidence deletion: evidence_id=%s error=%s", evidence_id, type(exc).__name__)
    try:
        ref.update({
            "lifecycle_status": "deleted",
            "deleted_at": datetime.now(timezone.utc).isoformat(),
            "deleted_by": user["uid"],
            "storage_path": None,
        })
        _refresh_evidence_count(case_id)
        _mark_index_stale(case_id, "evidence_deleted")
    except Exception:
        logger.exception("Evidence deletion metadata reconciliation required: evidence_id=%s", evidence_id)
        raise HTTPException(status_code=503, detail="Evidence file was deleted, but metadata reconciliation is required.") from None
    return {"evidence_id": evidence_id, "status": "deleted"}


@router.post("/{case_id}/evidence/reconcile")
def reconcile_evidence(case_id: str, user=Depends(get_current_user)):
    """Repair derived evidence counts and report interrupted/orphaned storage state."""
    check_case_access(case_id, user["uid"])
    records = database_ref(f"cases/{case_id}/evidence").get() or {}
    if not isinstance(records, dict):
        records = {}
    active_count = 0
    referenced_paths = set()
    missing = []
    interrupted = []
    for key, record in records.items():
        if not isinstance(record, dict):
            continue
        state = record.get("lifecycle_status", "active")
        path = record.get("storage_path")
        if state in {"active", "deleting", "storage_missing"}:
            active_count += 1
            if path:
                referenced_paths.add(str(path).replace("\\", "/"))
                try:
                    exists = safe_object_path(str(path), root=EVIDENCE_ROOT).is_file()
                except ValueError:
                    exists = False
                if not exists:
                    missing.append(key)
                    try:
                        database_ref(f"cases/{case_id}/evidence/{key}").update({"lifecycle_status": "storage_missing", "storage_error": "referenced_file_missing"})
                    except Exception:
                        logger.exception("Could not mark missing evidence: evidence_id=%s", key)
        if state in {"deleting", "uploading", "stored"}:
            interrupted.append({"evidence_id": key, "lifecycle_status": state})
    case_storage = EVIDENCE_ROOT / "cases" / case_id / "evidence"
    orphaned = []
    if case_storage.exists():
        for path in case_storage.rglob("original"):
            try:
                relative = path.resolve().relative_to(EVIDENCE_ROOT).as_posix()
            except ValueError:
                continue
            if relative not in referenced_paths:
                orphaned.append(relative)
    database_ref(f"cases/{case_id}").update({"evidence_count": active_count})
    return {"case_id": case_id, "evidence_count": active_count, "missing_files": missing, "orphaned_files": orphaned, "interrupted_records": interrupted, "orphan_cleanup": "manual_review_required"}


@router.post("/{case_id}/analyze")
def analyze_case(case_id: str, user=Depends(get_current_user)):
    """Backward-compatible alias for the single Hybrid RAG ingestion pipeline."""
    check_case_access(case_id, user["uid"])
    records = _active_evidence(case_id)
    if not records:
        raise HTTPException(status_code=400, detail="Upload evidence before starting analysis.")
    case_ref = database_ref(f"cases/{case_id}")
    case_ref.update({"status": "indexing"})
    try:
        result = ingest_case(case_id, records)
        for evidence_id, record in records.items():
            update = {
                "analysis_status": record.get("analysis_status", "failed"),
                "analysis": record.get("analysis"),
                "analysis_error": record.get("analysis_error"),
                "graph_status": record.get("graph_status", "unknown"),
            }
            database_ref(f"cases/{case_id}/evidence/{evidence_id}").update(update)
        database_ref(f"cases/{case_id}/analysis").update(
            analysis_record_from_result(
                result,
                ingested_evidence_ids=records.keys(),
                live_evidence_ids=_active_evidence(case_id).keys(),
            )
        )
        case_ref.update({"status": "analysis_completed" if result.get("published") else "analysis_partial"})
        return result
    except Exception as exc:
        logger.exception("Hybrid evidence ingestion failed: case_id=%s", case_id)
        try:
            case_ref.update({"status": "analysis_failed"})
        except Exception:
            logger.exception("Could not persist failed case status: case_id=%s", case_id)
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=500, detail="Investigation indexing failed. Check server logs.") from None
