import logging
from typing import Callable, Dict, Any

from ert.ensemble_evaluator import state

logger = logging.getLogger(__name__)

_handle = Callable[..., Any]


class EnsembleStateTracker:
    def __init__(self, state_: str = state.ENSEMBLE_STATE_UNKNOWN) -> None:
        self._state = state_
        self._handles: Dict[str, _handle] = {}
        self._msg = "Illegal state transition from %s to %s"

        self.set_default_handles()

    def add_handle(self, state_: str, handle: _handle) -> None:
        self._handles[state_] = handle

    def _handle_unknown(self) -> None:
        if self._state != state.ENSEMBLE_STATE_UNKNOWN:
            logger.warning(self._msg, self._state, state.ENSEMBLE_STATE_UNKNOWN)
        self._state = state.ENSEMBLE_STATE_UNKNOWN

    def _handle_started(self) -> None:
        if self._state != state.ENSEMBLE_STATE_UNKNOWN:
            logger.warning(self._msg, self._state, state.ENSEMBLE_STATE_STARTED)
        self._state = state.ENSEMBLE_STATE_STARTED

    def _handle_failed(self) -> None:
        if self._state not in [
            state.ENSEMBLE_STATE_UNKNOWN,
            state.ENSEMBLE_STATE_STARTED,
        ]:
            logger.warning(self._msg, self._state, state.ENSEMBLE_STATE_FAILED)
        self._state = state.ENSEMBLE_STATE_FAILED

    def _handle_stopped(self) -> None:
        if self._state != state.ENSEMBLE_STATE_STARTED:
            logger.warning(self._msg, self._state, state.ENSEMBLE_STATE_STOPPED)
        self._state = state.ENSEMBLE_STATE_STOPPED

    def _handle_canceled(self) -> None:
        if self._state != state.ENSEMBLE_STATE_STARTED:
            logger.warning(self._msg, self._state, state.ENSEMBLE_STATE_CANCELLED)
        self._state = state.ENSEMBLE_STATE_CANCELLED

    def set_default_handles(self) -> None:
        self.add_handle(state.ENSEMBLE_STATE_UNKNOWN, self._handle_unknown)
        self.add_handle(state.ENSEMBLE_STATE_STARTED, self._handle_started)
        self.add_handle(state.ENSEMBLE_STATE_FAILED, self._handle_failed)
        self.add_handle(state.ENSEMBLE_STATE_STOPPED, self._handle_stopped)
        self.add_handle(state.ENSEMBLE_STATE_CANCELLED, self._handle_canceled)

    def update_state(self, state_: str) -> str:
        if state_ not in self._handles:
            raise KeyError(f"Handle not defined for state {state_}")

        # Call the state handle mapped to the new state
        self._handles[state_]()

        return self._state
