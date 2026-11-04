"""Neo4j persistence and case-scoped graph retrieval for Mr. Sherlock.

Graph nodes represent machine-extracted candidates and source observations, not
verified forensic facts. Every returned chunk carries exhibit/chunk provenance.
"""
from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Any

_log = logging.getLogger(__name__)


def _enabled() -> bool:
    return os.getenv("SHERLOCK_ENABLE_NEO4J", "true").strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_driver():
    if not _enabled():
        raise RuntimeError("Neo4j integration is disabled.")
    uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    if not password:
        raise RuntimeError("NEO4J_PASSWORD is not configured.")
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(uri, auth=(username, password))
    driver.verify_connectivity()
    return driver


def close_driver() -> None:
    driver = get_driver.cache_info()
    if driver.currsize:
        instance = get_driver()
        instance.close()
        get_driver.cache_clear()


def ensure_schema() -> None:
    driver = get_driver()
    statements = [
        "CREATE CONSTRAINT investigation_case_id IF NOT EXISTS FOR (n:InvestigationCase) REQUIRE n.case_id IS UNIQUE",
        "CREATE CONSTRAINT evidence_identity IF NOT EXISTS FOR (n:Evidence) REQUIRE (n.case_id, n.evidence_id) IS UNIQUE",
        "CREATE CONSTRAINT chunk_identity IF NOT EXISTS FOR (n:EvidenceChunk) REQUIRE (n.case_id, n.chunk_id) IS UNIQUE",
        "CREATE CONSTRAINT candidate_entity_identity IF NOT EXISTS FOR (n:CandidateEntity) REQUIRE (n.case_id, n.normalized_name) IS UNIQUE",
        "CREATE INDEX candidate_entity_name IF NOT EXISTS FOR (n:CandidateEntity) ON (n.case_id, n.normalized_name)",
    ]
    with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
        for statement in statements:
            session.run(statement).consume()


def normalize_entity(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()[:250]


def question_terms(question: str) -> list[str]:
    # Include identifiers, dates, times and meaningful words; bounded to keep Cypher cheap.
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9:_-]{2,}", question or "")
    stop = {"the", "and", "for", "from", "with", "that", "this", "between", "what", "when", "where", "which", "show", "find", "tell", "about"}
    return list(dict.fromkeys(t.casefold() for t in tokens if t.casefold() not in stop))[:30]


def upsert_evidence_chunks(case_id: str, evidence_id: str, chunks: list[dict[str, Any]]) -> bool:
    """Index evidence chunks into Neo4j; logs and skips on Neo4j unavailability (H4).

    Returns True only when the graph write actually succeeded. Neo4j outages are
    swallowed so ingest survives, so the caller cannot rely on an exception to
    detect a degraded graph; the return value carries that signal instead.
    """
    if not chunks:
        return True
    try:
        ensure_schema()
        rows = []
        for item in chunks:
            entities = []
            for entity in item.get("entities", [])[:100]:
                raw = str(entity).strip()[:250]
                normalized = normalize_entity(raw)
                if raw and normalized:
                    entities.append({"name": raw, "normalized_name": normalized})
            rows.append({
                "chunk_id": str(item["chunk_id"]), "filename": str(item.get("filename", "")),
                "modality": str(item.get("modality", "unknown")),
                "text": str(item.get("text", ""))[:4000], "sha256": str(item.get("sha256", "")),
                "entities": entities,
            })
        driver = get_driver()
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            # Re-indexing can change chunk boundaries, so the exhibit's previous
            # chunks and co-mention edges are removed first. Otherwise orphaned
            # chunk nodes survive and graph retrieval would return text that no
            # longer exists in the authoritative FAISS generation.
            session.run("""
                MATCH (old:EvidenceChunk {case_id:$case_id, evidence_id:$evidence_id})
                DETACH DELETE old
            """, case_id=case_id, evidence_id=evidence_id).consume()
            session.run("""
                MATCH ()-[r:CO_MENTIONED_IN {case_id:$case_id, evidence_id:$evidence_id}]->()
                DELETE r
            """, case_id=case_id, evidence_id=evidence_id).consume()
            session.run("""
                MERGE (c:InvestigationCase {case_id:$case_id})
                MERGE (e:Evidence {case_id:$case_id, evidence_id:$evidence_id})
                SET e.sha256=$sha256, e.filename=$filename, e.modality=$modality,
                    e.lifecycle_status='active'
                MERGE (c)-[:HAS_EVIDENCE]->(e)
                WITH e
                UNWIND $rows AS row
                MERGE (ch:EvidenceChunk {case_id:$case_id, chunk_id:row.chunk_id})
                SET ch.text=row.text, ch.evidence_id=$evidence_id,
                    ch.filename=row.filename, ch.modality=row.modality,
                    ch.sha256=row.sha256, ch.verification_status='unverified'
                MERGE (e)-[:HAS_CHUNK]->(ch)
                WITH ch, row
                UNWIND row.entities AS entity
                MERGE (n:CandidateEntity {case_id:$case_id, normalized_name:entity.normalized_name})
                SET n.name=entity.name, n.verification_status='unverified'
                MERGE (ch)-[:MENTIONS {verification_status:'unverified'}]->(n)
            """, case_id=case_id, evidence_id=evidence_id, rows=rows,
                 sha256=rows[0]["sha256"], filename=rows[0]["filename"], modality=rows[0]["modality"]).consume()
            # A graph relationship means co-occurrence in a source chunk only; it is not
            # interpreted as a real-world relationship without independent verification.
            session.run("""
                MATCH (ch:EvidenceChunk {case_id:$case_id, evidence_id:$evidence_id})
                      -[:MENTIONS]->(a:CandidateEntity)
                MATCH (ch)-[:MENTIONS]->(b:CandidateEntity)
                WHERE a.normalized_name < b.normalized_name
                MERGE (a)-[r:CO_MENTIONED_IN {case_id:$case_id, chunk_id:ch.chunk_id}]->(b)
                SET r.verification_status='unverified', r.evidence_id=$evidence_id
            """, case_id=case_id, evidence_id=evidence_id).consume()
            # Drop entities that only existed in the removed chunks so the graph
            # does not accumulate stale vocabulary that could match a question.
            session.run("""
                MATCH (n:CandidateEntity {case_id:$case_id})
                WHERE NOT (n)<-[:MENTIONS]-(:EvidenceChunk)
                DETACH DELETE n
            """, case_id=case_id).consume()
        return True
    except Exception as exc:
        # H4: Neo4j outage must not crash the ingest pipeline. Graph enrichment is
        # supplementary; FAISS remains the authoritative retrieval store.
        _log.warning("graph_rag.upsert_evidence_chunks: Neo4j unavailable (%s: %s); skipping graph index.",
                     type(exc).__name__, exc)
        return False


def set_case_generation(case_id: str, generation: str, manifest_hash: str) -> bool:
    """Persist generation marker; logs and skips on Neo4j unavailability (H4).

    Returns True only when the marker was actually written, so callers can report
    an honest graph status instead of assuming success.
    """
    try:
        ensure_schema()
        with get_driver().session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            session.run("""
                MERGE (c:InvestigationCase {case_id:$case_id})
                SET c.active_generation=$generation, c.evidence_manifest_sha256=$manifest_hash,
                    c.graph_status='ready'
            """, case_id=case_id, generation=generation, manifest_hash=manifest_hash).consume()
        return True
    except Exception as exc:
        _log.warning("graph_rag.set_case_generation: Neo4j unavailable (%s: %s); generation not persisted.",
                     type(exc).__name__, exc)
        return False


def delete_evidence_graph(case_id: str, evidence_id: str) -> None:
    """Remove evidence graph nodes; logs and skips on Neo4j unavailability (H4)."""
    if not _enabled():
        return
    try:
        with get_driver().session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            session.run("""
                MATCH ()-[r:CO_MENTIONED_IN {case_id:$case_id, evidence_id:$evidence_id}]->()
                DELETE r
            """, case_id=case_id, evidence_id=evidence_id).consume()
            session.run("""
                MATCH (e:Evidence {case_id:$case_id, evidence_id:$evidence_id})
                OPTIONAL MATCH (e)-[:HAS_CHUNK]->(ch:EvidenceChunk)
                DETACH DELETE ch, e
            """, case_id=case_id, evidence_id=evidence_id).consume()
            session.run("""
                MATCH (n:CandidateEntity {case_id:$case_id})
                WHERE NOT (n)<-[:MENTIONS]-(:EvidenceChunk)
                DETACH DELETE n
            """, case_id=case_id).consume()
            session.run("""
                MATCH (c:InvestigationCase {case_id:$case_id})
                SET c.graph_status='stale'
            """, case_id=case_id).consume()
    except Exception as exc:
        _log.warning("graph_rag.delete_evidence_graph: Neo4j unavailable (%s: %s); graph nodes not cleaned up.",
                     type(exc).__name__, exc)


def retrieve_graph_context(case_id: str, question: str, limit: int = 8, expected_generation: str | None = None) -> dict[str, Any]:
    """Retrieve graph-linked chunks; returns a disabled/unavailable sentinel on failure (H4)."""
    if not _enabled():
        return {"status": "disabled", "relationships": [], "chunks": []}
    try:
        terms = question_terms(question)
        driver = get_driver()
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            state_record = session.run("MATCH (c:InvestigationCase {case_id:$case_id}) RETURN c.active_generation AS generation, c.graph_status AS status", case_id=case_id).single()
            if state_record is None:
                return {"status": "not_found", "relationships": [], "chunks": []}
            state = state_record.data()
            if expected_generation and state.get("generation") != expected_generation:
                return {"status": "stale", "relationships": [], "chunks": []}
            if state.get("status") == "stale":
                return {"status": "stale", "relationships": [], "chunks": []}
            if not terms:
                return {"status": state.get("status") or "ready", "relationships": [], "chunks": []}
            result = session.run("""
                MATCH (c:InvestigationCase {case_id:$case_id})
                OPTIONAL MATCH (c)-[:HAS_EVIDENCE]->(e:Evidence)
                WITH c, collect(DISTINCT e.evidence_id) AS current_evidence
                WITH c, current_evidence
                WHERE $expected_generation IS NULL OR c.active_generation = $expected_generation
                MATCH (n:CandidateEntity {case_id:$case_id})
                WHERE any(term IN $terms WHERE
                    n.normalized_name CONTAINS term OR term CONTAINS n.normalized_name)
                MATCH (ch:EvidenceChunk {case_id:$case_id})-[:MENTIONS]->(n)
                WHERE ch.evidence_id IN current_evidence
                OPTIONAL MATCH (ch)-[:MENTIONS]->(related:CandidateEntity)
                RETURN ch.evidence_id AS evidence_id, ch.chunk_id AS chunk_id,
                       ch.filename AS filename, ch.modality AS modality,
                       ch.sha256 AS sha256, ch.text AS text,
                       collect(DISTINCT related.name) AS entities,
                       collect(DISTINCT n.name) AS matched_entities,
                       c.active_generation AS active_generation,
                       c.graph_status AS graph_status
                ORDER BY ch.filename, ch.chunk_id LIMIT $limit
            """, case_id=case_id, terms=terms, expected_generation=expected_generation,
                 limit=max(1, min(int(limit), 24)))
            chunks, relationships, status = [], [], state.get("status") or "ready"
            for record in result:
                data = record.data()
                status = data.get("graph_status") or "ready"
                chunks.append({k: data.get(k) for k in ("evidence_id", "chunk_id", "filename", "modality", "sha256", "text")})
                relationships.append({
                    "chunk_id": data.get("chunk_id"), "evidence_id": data.get("evidence_id"),
                    "matched_entities": data.get("matched_entities") or [],
                    "co_mentioned_entities": data.get("entities") or [],
                    "relationship_type": "co_occurrence_in_same_evidence_chunk",
                    "verification_status": "unverified",
                })
        return {"status": status, "relationships": relationships, "chunks": chunks}
    except Exception as exc:
        # H4: Neo4j outage degrades to no graph context; FAISS answer path continues.
        _log.warning("graph_rag.retrieve_graph_context: Neo4j unavailable (%s: %s); returning empty context.",
                     type(exc).__name__, exc)
        return {"status": "unavailable", "error": type(exc).__name__, "relationships": [], "chunks": []}
