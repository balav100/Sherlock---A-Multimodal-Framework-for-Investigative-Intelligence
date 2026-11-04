import json
import pytest
from backend.services import hybrid_rag


CASE_ID = "CASE-TEST123456"


def _create_case_files(tmp_path, metadata, index_bytes=b"placeholder"):
    folder = tmp_path / CASE_ID
    folder.mkdir()
    (folder / "evidence.faiss").write_bytes(index_bytes)
    (folder / "chunks.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    return folder


def _valid_chunk():
    return {
        "case_id": CASE_ID,
        "evidence_id": "E001",
        "chunk_id": "C001",
        "filename": "evidence.txt",
        "text": "A sample evidence statement.",
    }


def test_answer_case_rejects_malformed_json_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(hybrid_rag, "INDEX_ROOT", tmp_path)
    folder = tmp_path / CASE_ID
    folder.mkdir()
    (folder / "evidence.faiss").write_bytes(b"placeholder")
    (folder / "chunks.json").write_text(
        '{"broken": ',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="metadata is unreadable"):
        hybrid_rag.answer_case(CASE_ID, "Where is the evidence?")


def test_answer_case_rejects_metadata_that_is_not_a_list(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(hybrid_rag, "INDEX_ROOT", tmp_path)
    _create_case_files(tmp_path, {"unexpected": "object"})

    with pytest.raises(ValueError, match="metadata must be a list"):
        hybrid_rag.answer_case(CASE_ID, "Where is the evidence?")


def test_answer_case_rejects_metadata_with_missing_source_fields(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(hybrid_rag, "INDEX_ROOT", tmp_path)
    _create_case_files(
        tmp_path,
        [{
            "case_id": CASE_ID,
            "evidence_id": "E001",
        }],
    )

    with pytest.raises(ValueError, match="missing required source metadata"):
        hybrid_rag.answer_case(CASE_ID, "Where is the evidence?")


import sys
from types import SimpleNamespace


def test_answer_case_rejects_faiss_vector_count_mismatch(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(hybrid_rag, "INDEX_ROOT", tmp_path)
    _create_case_files(tmp_path, [_valid_chunk()])

    fake_index = SimpleNamespace(ntotal=2, d=384)
    fake_faiss = SimpleNamespace(
        read_index=lambda path: fake_index
    )
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)

    with pytest.raises(ValueError, match="vector count does not match"):
        hybrid_rag.answer_case(CASE_ID, "Where is the evidence?")


def test_answer_case_rejects_embedding_dimension_mismatch(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(hybrid_rag, "INDEX_ROOT", tmp_path)
    _create_case_files(tmp_path, [_valid_chunk()])

    fake_index = SimpleNamespace(ntotal=1, d=123)
    fake_faiss = SimpleNamespace(
        read_index=lambda path: fake_index
    )
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)

    fake_embedder = SimpleNamespace(
        get_sentence_embedding_dimension=lambda: 384
    )
    monkeypatch.setattr(
        hybrid_rag, "_embedder", lambda: fake_embedder
    )

    with pytest.raises(ValueError, match="does not match"):
        hybrid_rag.answer_case(CASE_ID, "Where is the evidence?")
