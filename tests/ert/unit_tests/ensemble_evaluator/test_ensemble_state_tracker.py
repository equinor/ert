import logging

import pytest

from ert.ensemble_evaluator import state
from ert.ensemble_evaluator._ensemble import _EnsembleStateTracker


@pytest.mark.parametrize(
    ("transition", "allowed"),
    [
        ([state.ENSEMBLE_STATE_STARTED, state.ENSEMBLE_STATE_STOPPED], True),
        ([state.ENSEMBLE_STATE_STARTED, state.ENSEMBLE_STATE_FAILED], True),
        ([state.ENSEMBLE_STATE_STARTED, state.ENSEMBLE_STATE_CANCELLED], True),
        ([state.ENSEMBLE_STATE_CANCELLED, state.ENSEMBLE_STATE_STARTED], False),
        ([state.ENSEMBLE_STATE_CANCELLED, state.ENSEMBLE_STATE_STOPPED], False),
        ([state.ENSEMBLE_STATE_CANCELLED, state.ENSEMBLE_STATE_FAILED], False),
        ([state.ENSEMBLE_STATE_STOPPED, state.ENSEMBLE_STATE_FAILED], False),
        ([state.ENSEMBLE_STATE_STOPPED, state.ENSEMBLE_STATE_CANCELLED], False),
        ([state.ENSEMBLE_STATE_STOPPED, state.ENSEMBLE_STATE_STARTED], False),
        ([state.ENSEMBLE_STATE_FAILED, state.ENSEMBLE_STATE_STARTED], False),
        ([state.ENSEMBLE_STATE_FAILED, state.ENSEMBLE_STATE_STOPPED], False),
        ([state.ENSEMBLE_STATE_FAILED, state.ENSEMBLE_STATE_CANCELLED], False),
        ([state.ENSEMBLE_STATE_UNKNOWN, state.ENSEMBLE_STATE_STARTED], True),
    ],
)
def test_ensemble_state_tracker(transition, allowed, caplog, snapshot):
    initial_state, update_state = transition
    with caplog.at_level(logging.WARNING):
        state_tracker = _EnsembleStateTracker(initial_state)
        new_state = state_tracker.update_state(update_state)
        assert new_state == update_state
        if allowed:
            assert len(caplog.messages) == 0
        else:
            assert len(caplog.messages) == 1
            log_mgs = f"Illegal state transition from {initial_state} to {update_state}"
            assert log_mgs == caplog.messages[0]


def test_ensemble_state_tracker_handles():
    state_machine = _EnsembleStateTracker()
    expected_sates = [
        state.ENSEMBLE_STATE_UNKNOWN,
        state.ENSEMBLE_STATE_STARTED,
        state.ENSEMBLE_STATE_FAILED,
        state.ENSEMBLE_STATE_STOPPED,
        state.ENSEMBLE_STATE_CANCELLED,
    ]
    handled_states = list(state_machine._handles.keys())
    assert len(handled_states) == len(expected_sates)
    for handled_state in handled_states:
        assert handled_state in expected_sates
