import json
import pytest
from backend.services import hybrid_rag


def test_answer_case_rejects_missing_index_without_rebuilding(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(hybrid_rag, "INDEX_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError, match="No complete evidence index"):
        hybrid_rag.answer_case("CASE-TEST123456", "Where is the evidence?")

    case_folder = tmp_path / "CASE-TEST123456"
    assert not (case_folder / "evidence.faiss").exists()
    assert not (case_folder / "chunks.json").exists()


def test_answer_case_rejects_metadata_from_another_case(
    monkeypatch, tmp_path
):
    case_id = "CASE-TEST123456"
    monkeypatch.setattr(hybrid_rag, "INDEX_ROOT", tmp_path)

    case_folder = tmp_path / case_id
    case_folder.mkdir()

    # Create placeholder files; metadata validation occurs before FAISS loading.
    (case_folder / "evidence.faiss").write_bytes(b"test placeholder")
    (case_folder / "chunks.json").write_text(
        json.dumps([
            {
                "case_id": "CASE-DIFFERENT",
                "evidence_id": "E001",
                "chunk_id": "C001",
                "filename": "evidence.txt",
                "text": "Test evidence content",
            }
        ]),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="records from another case"
    ):
        hybrid_rag.answer_case(case_id, "Where is the evidence?")
