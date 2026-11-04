from backend.domain.processing_states import (
    COMPLETED,
    EXTRACTING,
    FAILED,
    InvalidStateTransition,
    QUEUED,
    UPLOADED,
    can_transition,
    transition,
)
import pytest


def test_upload_to_queued_is_legal():
    assert transition(UPLOADED, QUEUED) == QUEUED
    assert can_transition(UPLOADED, QUEUED)


def test_completed_cannot_go_directly_to_extracting():
    with pytest.raises(InvalidStateTransition):
        transition(COMPLETED, EXTRACTING)


def test_failed_can_be_requeued():
    assert transition(FAILED, QUEUED) == QUEUED


def test_same_state_is_idempotent():
    assert transition(EXTRACTING, EXTRACTING) == EXTRACTING
