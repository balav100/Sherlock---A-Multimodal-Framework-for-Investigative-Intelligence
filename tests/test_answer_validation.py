import pytest

from backend.services import hybrid_rag


@pytest.mark.parametrize(
    "case_id",
    ["", "   ", None],
)
def test_answer_case_rejects_invalid_case_id(case_id):
    with pytest.raises(ValueError, match="Case ID cannot be empty"):
        hybrid_rag.answer_case(case_id, "Where is the evidence?")


@pytest.mark.parametrize(
    "question",
    ["", "  ", "ab"],
)
def test_answer_case_rejects_short_question(question):
    with pytest.raises(ValueError, match="at least 3 characters"):
        hybrid_rag.answer_case("CASE-TEST123456", question)


def test_answer_case_rejects_question_over_2000_characters():
    with pytest.raises(ValueError, match="2000 characters"):
        hybrid_rag.answer_case("CASE-TEST123456", "x" * 2001)


@pytest.mark.parametrize(
    "top_k",
    [0, -1, 13],
)
def test_answer_case_rejects_out_of_range_top_k(top_k):
    with pytest.raises(ValueError, match="between 1 and 12"):
        hybrid_rag.answer_case(
            "CASE-TEST123456",
            "Where is the evidence?",
            top_k=top_k,
        )


@pytest.mark.parametrize(
    "top_k",
    [True, False, 1.5, "6"],
)
def test_answer_case_rejects_non_integer_top_k(top_k):
    with pytest.raises(ValueError, match="top_k must be an integer"):
        hybrid_rag.answer_case(
            "CASE-TEST123456",
            "Where is the evidence?",
            top_k=top_k,
        )
