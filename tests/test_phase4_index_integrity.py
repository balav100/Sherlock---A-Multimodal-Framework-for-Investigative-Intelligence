from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from backend.services import hybrid_rag


class FakeEmbedder:
    def encode(self, chunks, **kwargs):
        return np.asarray([[float(len(text)), 1.0] for text in chunks], dtype="float32")


def _record(root: Path, name: str, content: bytes, evidence_id: str = "E1"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return evidence_id, {
        "case_id": "CASE1",
        "storage_path": name,
        "original_filename": name,
        "modality": "txt",
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _configure(monkeypatch, tmp_path):
    evidence_root = tmp_path / "evidence"
    index_root = tmp_path / "indexes"
    monkeypatch.setattr(hybrid_rag, "EVIDENCE_ROOT", evidence_root)
    monkeypatch.setattr(hybrid_rag, "INDEX_ROOT", index_root)
    monkeypatch.setattr(hybrid_rag, "CACHE_ROOT", index_root / "_cache")
    monkeypatch.setattr(hybrid_rag, "_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(hybrid_rag, "_extract", lambda path, filename: (path.read_text(), []))
    monkeypatch.setattr(hybrid_rag, "_write_graph", lambda **kwargs: None)
    return evidence_root, index_root


def test_incremental_ingest_reuses_content_addressed_extraction_and_embeddings(monkeypatch, tmp_path):
    evidence_root, index_root = _configure(monkeypatch, tmp_path)
    calls = {"extract": 0}

    def extract(path, filename):
        calls["extract"] += 1
        return path.read_text(), []

    monkeypatch.setattr(hybrid_rag, "_extract", extract)
    publish_calls = []

    def publish(case_id, vectors, metadata, manifest):
        publish_calls.append((vectors, metadata, manifest))
        return {"generation": f"gen-{len(publish_calls)}", "manifest": manifest, "directory": str(index_root / "published")}

    monkeypatch.setattr(hybrid_rag, "_publish_generation", publish)
    evidence_id, record = _record(evidence_root, "one.txt", b"door opened at 09:10")

    first = hybrid_rag.ingest_case("CASE1", {evidence_id: record.copy()})
    second = hybrid_rag.ingest_case("CASE1", {evidence_id: record.copy()})

    assert first["published"] is True
    assert second["published"] is True
    assert calls["extract"] == 1
    assert len(publish_calls) == 2
    assert first["manifest"]["evidence_manifest_sha256"] == second["manifest"]["evidence_manifest_sha256"]


def test_partial_failure_does_not_publish_or_replace_previous_generation(monkeypatch, tmp_path):
    evidence_root, index_root = _configure(monkeypatch, tmp_path)
    case_dir = index_root / "CASE1"
    case_dir.mkdir(parents=True)
    (case_dir / "active.json").write_text(json.dumps({"generation": "known-good"}), encoding="utf-8")
    publish = []
    monkeypatch.setattr(hybrid_rag, "_publish_generation", lambda *args, **kwargs: publish.append(True))

    evidence_id, good = _record(evidence_root, "good.txt", b"valid evidence", "E1")
    bad_id, bad = _record(evidence_root, "bad.txt", b"tampered evidence", "E2")
    bad["sha256"] = "0" * 64

    result = hybrid_rag.ingest_case("CASE1", {evidence_id: good, bad_id: bad})

    assert result["published"] is False
    assert result["preserved_previous_generation"] is True
    assert publish == []
    assert json.loads((case_dir / "active.json").read_text()) == {"generation": "known-good"}


def test_active_generation_pointer_rejects_path_traversal(monkeypatch, tmp_path):
    _, index_root = _configure(monkeypatch, tmp_path)
    case_dir = index_root / "CASE1"
    case_dir.mkdir(parents=True)
    (case_dir / "active.json").write_text(json.dumps({"generation": "../escape"}), encoding="utf-8")

    try:
        hybrid_rag._active_generation("CASE1")
    except ValueError as exc:
        assert "pointer is invalid" in str(exc)
    else:
        raise AssertionError("Traversal pointer should be rejected")
