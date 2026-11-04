from __future__ import annotations

import json

import pytest

from backend.services import hybrid_rag


def test_timed_stage_emits_machine_readable_duration(capsys):
    with hybrid_rag._timed_stage("unit_test_stage", case_id="CASE-TEST"):
        pass

    output = capsys.readouterr().out.strip()
    assert output.startswith("[INDEX_TIMING] ")
    payload = json.loads(output.split(" ", 1)[1])
    assert payload["stage"] == "unit_test_stage"
    assert payload["case_id"] == "CASE-TEST"
    assert payload["status"] == "ok"
    assert payload["seconds"] >= 0


def test_timed_stage_records_failure_and_reraises(capsys):
    with pytest.raises(ValueError, match="expected"):
        with hybrid_rag._timed_stage("failing_stage"):
            raise ValueError("expected")

    output = capsys.readouterr().out.strip()
    payload = json.loads(output.split(" ", 1)[1])
    assert payload["stage"] == "failing_stage"
    assert payload["status"] == "error"
