import asyncio
import functools
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from ert.run_models import StatusEvents


class StatusEventNotifier(Protocol):
    def __call__(self, *, experiment_run_id: str, event: StatusEvents) -> None: ...


class StatusEventListener(Protocol):
    def __call__(self, *, event: StatusEvents) -> None: ...


_status_event_listener_executor = ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="status_event_listener"
)


class StatusEventForwarder:
    """Passes status events to a listener one at a time, in the order they
    were forwarded, without blocking the event loop.

    An event is pending until the listener has returned from handling it.
    """

    def __init__(self, listener: StatusEventListener) -> None:
        self._listener = listener
        self._events: deque[StatusEvents] = deque()
        self._delivery: asyncio.Future[None] | None = None

    @property
    def pending(self) -> int:
        return len(self._events)

    def forward(self, event: StatusEvents) -> None:
        """Must be called from the event loop thread."""
        self._events.append(event)
        if self._delivery is None:
            self._deliver_next()

    def _deliver_next(self) -> None:
        delivery = asyncio.get_running_loop().run_in_executor(
            _status_event_listener_executor,
            functools.partial(self._listener, event=self._events[0]),
        )
        self._delivery = delivery
        delivery.add_done_callback(self._on_delivered)

    def _on_delivered(self, delivery: asyncio.Future[None]) -> None:
        self._events.popleft()
        self._delivery = None
        if not delivery.cancelled() and (error := delivery.exception()) is not None:
            logging.getLogger(__name__).error(
                "Status event listener failed to handle event", exc_info=error
            )
        if self._events:
            self._deliver_next()


def create_status_event_forwarder(
    notifier: StatusEventNotifier | None, experiment_run_id: str
) -> StatusEventForwarder | None:
    if notifier is None:
        return None
    return StatusEventForwarder(
        functools.partial(notifier, experiment_run_id=experiment_run_id)
    )
