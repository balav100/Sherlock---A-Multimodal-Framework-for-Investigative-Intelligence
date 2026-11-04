from __future__ import annotations

import json
from pathlib import Path

from backend.services import hybrid_rag


def test_observation_schema_has_provenance_and_unverified_status():
    item = hybrid_rag._observation(
        "audio_transcript_segment",
        {"start_seconds": 1.25, "end_seconds": 2.5},
        "A door closes.",
        "faster_whisper:base",
    )
    assert item["schema_version"] == 1
    assert item["source_location"]["start_seconds"] == 1.25
    assert item["verification_status"] == "unverified_machine_observation"
    assert item["processing_status"] == "extracted"


def test_text_extraction_emits_source_observation(tmp_path: Path):
    source = tmp_path / "notes.txt"
    source.write_text("Witness arrived at 09:10.", encoding="utf-8")
    text, observations = hybrid_rag._extract(source, source.name)
    assert "09:10" in text
    assert observations[0]["observation_type"] == "text_document"
    assert observations[0]["source_location"]["line_range"] == "whole_file"


def test_csv_extraction_emits_tabular_observation(tmp_path: Path):
    source = tmp_path / "log.csv"
    source.write_text("time,event\n09:10,door opened\n", encoding="utf-8")
    text, observations = hybrid_rag._extract(source, source.name)
    assert "door opened" in text
    assert observations[0]["observation_type"] == "tabular_text"


def test_json_extraction_validates_and_emits_root_observation(tmp_path: Path):
    source = tmp_path / "record.json"
    source.write_text(json.dumps({"employee": "E-17"}), encoding="utf-8")
    text, observations = hybrid_rag._extract(source, source.name)
    assert '"E-17"' in text
    assert observations[0]["source_location"]["json_path"] == "$"


def test_unsupported_extension_fails_closed(tmp_path: Path):
    source = tmp_path / "unknown.bin"
    source.write_bytes(b"abc")
    try:
        hybrid_rag._extract(source, source.name)
    except ValueError as exc:
        assert "Unsupported evidence type" in str(exc)
    else:
        raise AssertionError("Unsupported evidence type should fail")
