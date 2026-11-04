from backend.services.answer_engine import (
    build_claims, correlate_claims, detect_contradictions,
    generate_answer, verify_claims,
)


def chunk(eid, cid, text, modality="text", **extra):
    return {
        "case_id": "CASE-1", "evidence_id": eid, "chunk_id": cid,
        "filename": f"{eid}.txt", "sha256": "a" * 64,
        "modality": modality, "text": text, **extra,
    }


def test_claims_preserve_distinct_exhibit_provenance_even_for_same_sentence():
    chunks = [chunk("E1", "C1", "Employee EMP-104 entered the storage room."),
              chunk("E2", "C2", "Employee EMP-104 entered the storage room.")]
    claims = build_claims(chunks)
    assert len(claims) == 2
    assert {c["source_refs"][0]["evidence_id"] for c in claims} == {"E1", "E2"}
    assert verify_claims(claims)["status"] == "passed"


def test_exact_identifier_match_is_candidate_not_verified_relationship():
    claims = build_claims([
        chunk("E1", "C1", "Badge EMP-104 was recorded at 09:10."),
        chunk("E2", "C2", "EMP-104 appears in the visitor register at 09:12."),
    ])
    links = correlate_claims(claims)
    assert len(links) == 1
    assert links[0]["match_type"] == "identifier_match"
    assert links[0]["verification_status"] == "unverified_candidate"
    assert len(links[0]["claim_ids"]) == 2


def test_shared_time_only_is_weak_and_never_proof():
    claims = build_claims([
        chunk("E1", "C1", "A motion event was logged at 09:10."),
        chunk("E2", "C2", "A door event was recorded at 09:10."),
    ])
    links = correlate_claims(claims)
    assert links and links[0]["match_type"] == "time_match"
    assert links[0]["lead_strength"] < 0.5


def test_contradiction_rule_requires_same_anchor_across_exhibits():
    claims = build_claims([
        chunk("E1", "C1", "Door D-12 was open during inspection."),
        chunk("E2", "C2", "Door D-12 was closed during inspection."),
    ])
    conflicts = detect_contradictions(claims)
    assert len(conflicts) == 1
    assert conflicts[0]["status"] == "potential_conflict_requires_review"


def test_answer_abstains_when_no_question_relevance():
    result = generate_answer("What was the weather on Mars?", [
        chunk("E1", "C1", "Employee EMP-104 entered the storage room at 09:10.")
    ])
    assert result["answer_status"] == "abstained_low_relevance"
    assert "abstain" in result["answer"].lower() or "could not" in result["answer"].lower()


def test_answer_contains_claim_level_source_citation_and_provenance_validation():
    result = generate_answer("Where did EMP-104 enter?", [
        chunk("E1", "C1", "Employee EMP-104 entered the storage room at 09:10.", page=3)
    ])
    assert result["answer_status"] == "answered_with_sources"
    assert result["provenance_validation"]["status"] == "passed"
    assert "E1" in result["answer"] and "C1" in result["answer"]
    assert result["claims"][0]["source_refs"][0]["page"] == 3


def test_visual_and_audio_claims_keep_observation_classification():
    claims = build_claims([
        chunk("E1", "C1", "A person wearing a dark jacket is visible.", "image"),
        chunk("E2", "C2", "The speaker says the door is open.", "audio"),
    ])
    assert claims[0]["classification"] == "VISION_OBSERVATION"
    assert claims[1]["classification"] == "TRANSCRIPTION"


def test_same_exhibit_does_not_create_cross_exhibit_link_or_conflict():
    claims = build_claims([
        chunk("E1", "C1", "Door D-12 was open during inspection."),
        chunk("E1", "C2", "Door D-12 was closed during inspection."),
    ])
    assert correlate_claims(claims) == []
    assert detect_contradictions(claims) == []


def test_verify_claims_rejects_hallucinated_chunk_citation():
    chunks = [chunk("E1", "C1", "Door D-12 was open.")]
    claims = build_claims(chunks)
    # Simulate a corrupted or hallucinated chunk_id in the citation
    claims[0]["source_refs"][0]["chunk_id"] = "FABRICATED-CHUNK-999"
    validation = verify_claims(claims, retrieved_chunks=chunks)
    assert validation["status"] == "failed"
    assert claims[0]["claim_id"] in validation["invalid_claim_ids"]


def test_expanded_contradictions_detects_access_and_presence_conflicts():
    claims = [
        {
            "claim_id": "CLM-1",
            "text": "Employee EMP-104 entry was granted at gate.",
            "source_refs": [{"evidence_id": "E1", "chunk_id": "C1", "filename": "log.txt", "excerpt": "..."}],
            "anchors": {"identifier": ["EMP-104"]},
        },
        {
            "claim_id": "CLM-2",
            "text": "Employee EMP-104 entry was denied at gate.",
            "source_refs": [{"evidence_id": "E2", "chunk_id": "C2", "filename": "security.txt", "excerpt": "..."}],
            "anchors": {"identifier": ["EMP-104"]},
        },
    ]
    conflicts = detect_contradictions(claims)
    assert len(conflicts) == 1
    assert conflicts[0]["subject_anchor"] == "EMP-104"
    assert conflicts[0]["topic"] == "access_state"
