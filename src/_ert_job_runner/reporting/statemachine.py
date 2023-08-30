import logging
from typing import Callable, Dict, Optional, Tuple, Type

from _ert_job_runner.reporting.message import (
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


State = Tuple[Type[Message], ...]


class StateMachine:
    def __init__(self) -> None:
        logger.debug("Initializing state machines")
        initialized = (Init,)
        jobs = (Start, Running, Exited)
        finished = (Finish,)
        self._handler: Dict[State, Callable[[Message], None]] = {}
        self._transitions: Dict[Optional[State], State] = {
            None: initialized,
            initialized: jobs + finished,
            jobs: jobs + finished,
        }
        self._state: Optional[State] = None

    def add_handler(self, states: State, handler: Callable[[Message], None]) -> None:
        if states in self._handler:
            raise ValueError(f"{states} already handled by {self._handler[states]}")
        self._handler[states] = handler

    def transition(self, message: Message) -> None:
        new_state = None
        for state in self._handler:
            if isinstance(message, state):
                new_state = state
        assert new_state is not None

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
