# Imports the canonical hybrid_rag module directly.
# (Previously pointed at a non-existent 'hybrid_rag_retrieval_fixed.py' -- C2 fix.)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.services import hybrid_rag as HYBRID


def test_connect_question_uses_cross_evidence_mode():
    assert HYBRID._is_cross_evidence_question(
        "What connects the CCTV footage to the asset checkout log?"
    )


def test_link_question_uses_cross_evidence_mode():
    assert HYBRID._is_cross_evidence_question(
        "What links the two exhibits?"
    )


def test_related_question_uses_cross_evidence_mode():
    assert HYBRID._is_cross_evidence_question(
        "How are these exhibits related?"
    )


def test_compare_question_uses_cross_evidence_mode():
    assert HYBRID._is_cross_evidence_question(
        "Compare the CCTV footage with the access log."
    )


def test_ordinary_question_does_not_force_cross_evidence_mode():
    assert not HYBRID._is_cross_evidence_question(
        "What time does the checkout log show?"
    )
