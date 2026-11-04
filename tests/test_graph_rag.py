from backend.services import graph_rag


def test_normalize_entity_is_case_and_whitespace_insensitive():
    assert graph_rag.normalize_entity("  EMP-104   Door A ") == "emp-104 door a"


def test_question_terms_preserve_identifiers_and_drop_stop_words():
    terms = graph_rag.question_terms("What connects EMP-104 with camera 09:10 on 2026-10-31?")
    assert "emp-104" in terms
    assert "09:10" in terms
    assert "2026-10-31" in terms
    assert "what" not in terms


def test_retrieval_refuses_graph_generation_mismatch(monkeypatch):
    class Record:
        def data(self):
            return {"generation": "old-generation", "status": "ready"}

    class Result:
        def single(self):
            return Record()

    class Session:
        def __init__(self):
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, *args, **kwargs):
            self.calls += 1
            return Result()

    session = Session()

    class Driver:
        def session(self, **kwargs):
            return session

    monkeypatch.setattr(graph_rag, "_enabled", lambda: True)
    monkeypatch.setattr(graph_rag, "get_driver", lambda: Driver())
    result = graph_rag.retrieve_graph_context(
        "CASE1", "Find EMP-104", expected_generation="new-generation"
    )
    assert result == {"status": "stale", "relationships": [], "chunks": []}
    assert session.calls == 1


def test_disabled_graph_returns_explicit_disabled_status(monkeypatch):
    monkeypatch.setattr(graph_rag, "_enabled", lambda: False)
    assert graph_rag.retrieve_graph_context("CASE1", "EMP-104") == {
        "status": "disabled", "relationships": [], "chunks": []
    }


def test_connection_failure_returns_unavailable_status(monkeypatch):
    monkeypatch.setattr(graph_rag, "_enabled", lambda: True)
    def fail_driver():
        raise RuntimeError("Connection refused")
    monkeypatch.setattr(graph_rag, "get_driver", fail_driver)
    result = graph_rag.retrieve_graph_context("CASE1", "EMP-104")
    assert result["status"] == "unavailable"
    assert result["error"] == "RuntimeError"


def test_upsert_connection_failure_does_not_raise(monkeypatch):
    def fail_schema():
        raise RuntimeError("Neo4j down")
    monkeypatch.setattr(graph_rag, "ensure_schema", fail_schema)
    # Must not raise — fails open/logs warning (H4)
    graph_rag.upsert_evidence_chunks("CASE1", "E1", [{"chunk_id": "C1", "text": "hello"}])
