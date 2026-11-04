from backend.services import investigation_engine as engine


def test_investigation_failure_does_not_expose_internal_details(monkeypatch):
    secret_path = "Internal path: C:/private/evidence/secret.txt"

    def fail_analysis(record):
        raise RuntimeError(secret_path)

    monkeypatch.setattr(engine, "analyze_evidence", fail_analysis)

    result = engine.run_investigation([
        {"evidence_id": "EVD-TEST123"}
    ])

    assert result["overall_status"] == "failed"
    assert result["failures"][0]["analysis_status"] == "failed"
    assert result["failures"][0]["error"] == (
        "Evidence analysis failed. Check server logs for details."
    )
    assert secret_path not in str(result)
