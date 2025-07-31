import logging
from collections.abc import Callable
from typing import Any

from _ert.forward_model_runner.reporting.message import (
    Checksum,
    Exited,
    Finish,
    Init,
    Message,
    Running,
    Start,
)

logger = logging.getLogger(__name__)


class TransitionError(ValueError):
    pass


class StateMachine:
    def __init__(self) -> None:
        logger.debug("Initializing state machines")
        initialized = (Init,)
        steps = (Start, Running, Exited)
        checksum = (Checksum,)
        finished = (Finish,)
        self._handler: dict[Any, Callable[[Any], None]] = {}
        self._transitions = {
            None: (*initialized, Exited),
            initialized: steps + checksum + finished,
            steps: steps + checksum + finished,
            checksum: checksum + finished,
        }
        self._state: Any = None

    def add_handler(
        self, states: tuple[type[Message], ...], handler: Callable[[Any], None]
    ) -> None:
        if states in self._handler:
            raise ValueError(f"{states} already handled by {self._handler[states]}")
        self._handler[states] = handler

    def transition(self, message: Any) -> None:
        new_state = None
        for state in self._handler:
            if isinstance(message, state):
                new_state = state

        if self._state not in self._transitions or not isinstance(
            message, self._transitions[self._state]
        ):
            logger.error(
                f"{message} illegal state transition: {self._state} -> {new_state}"
            )
            raise TransitionError(
                f"Illegal transition {self._state} -> {new_state} for {message}, "
                f"expected to transition into {self._transitions[self._state]}"
            )

        self._handler[new_state](message)
        self._state = new_state
