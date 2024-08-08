from __future__ import annotations

import logging
from contextlib import suppress
from queue import Empty, SimpleQueue
from time import sleep
from typing import Optional

from qtpy.QtCore import QObject, Signal, Slot

from ert.ensemble_evaluator import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)
from ert.gui.model.snapshot import SnapshotModel
from ert.run_models import StatusEvents

logger = logging.getLogger(__name__)


class QueueEmitter(QObject):
    """A worker that emits items put on a queue to qt subscribers."""

    new_event = Signal(object)
    done = Signal()

    def __init__(
        self,
        event_queue: SimpleQueue[StatusEvents],
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        logger.debug("init QueueEmitter")
        self._event_queue = event_queue
        self._stopped = False

    @Slot()
    def consume_and_emit(self) -> None:
        logger.debug("tracking...")
        while True:
            event = None
            with suppress(Empty):
                event = self._event_queue.get(timeout=1.0)
            if self._stopped:
                logger.debug("stopped")
                break
            if event is None:
                sleep(0.1)
                continue

            # pre-rendering in this thread to avoid work in main rendering thread
            if (
                isinstance(event, (FullSnapshotEvent, SnapshotUpdateEvent))
                and event.snapshot
            ):
                SnapshotModel.prerender(event.snapshot)

            logger.debug(f"emit {event}")
            self.new_event.emit(event)

            if isinstance(event, EndEvent):
                logger.debug("got end event")
                break

        self.done.emit()
        logger.debug("tracking done.")

    @Slot()
    def stop(self) -> None:
        logger.debug("stopping...")
        self._stopped = True
