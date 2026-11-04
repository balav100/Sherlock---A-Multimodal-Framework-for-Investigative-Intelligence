# Imports the canonical answer_engine module directly.
# (Previously pointed at a non-existent 'answer_engine_fixed.py' -- C2 fix.)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.services import answer_engine as engine


def _claim(claim_id, evidence_id, text, identifiers=None, times=None):
    return {
        "claim_id": claim_id,
        "text": text,
        "source_refs": [{"evidence_id": evidence_id}],
        "anchors": {
            "identifier": identifiers or [],
            "date": [],
            "time": times or [],
        },
    }


def test_correlation_cites_claims_that_produced_best_match():
    claims = [
        _claim("A-best", "E1", "Asset LPT-8876 checked out", ["LPT-8876"]),
        _claim("A-last", "E1", "Gate opened at 21:14", times=["21:14"]),
        _claim("B-best", "E2", "CCTV references LPT-8876", ["LPT-8876"]),
        _claim("B-last", "E2", "Gate recorded at 21:14", times=["21:14"]),
    ]
    links = engine.correlate_claims(claims)
    assert len(links) == 1
    assert links[0]["match_type"] == "identifier_match"
    assert links[0]["claim_ids"] == ["A-best", "B-best"]


def test_cctv_still_image_is_not_misclassified_as_video():
    assert engine._modality({
        "modality": "unknown",
        "filename": "CCTV_Log.png",
    }) == "image"


def test_cctv_video_is_classified_as_video():
    assert engine._modality({
        "modality": "unknown",
        "filename": "Warehouse_CCTV.mp4",
    }) == "video"
