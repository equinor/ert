from __future__ import annotations

import logging
from contextlib import suppress
from queue import Empty
from time import sleep
from typing import Optional

from qtpy.QtCore import QObject, Signal, Slot

from ert.ensemble_evaluator import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
    Snapshot
)
from ert.gui.model.snapshot import SnapshotModel
from ert.run_models import StatusEvents

from websockets.sync.client import connect

import json

logger = logging.getLogger(__name__)

from pydantic import BaseModel, ConfigDict

class EventWrapper(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    event: StatusEvents

class EventFetcher(QObject):
    """A worker that emits items put on a queue to qt subscribers."""

    new_event = Signal(object)
    done = Signal()

    def __init__(
        self,
        experiment_id: str,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        logger.debug("init EventFetcher")
        self._experiment_id = experiment_id
        self._stopped = False

    @Slot()
    def consume_and_emit(self) -> None:
        logger.debug("tracking...")
        with connect(f"ws://localhost:8000/experiments/{self._experiment_id}/events") as websocket:
            while True:
                try:
                    message = websocket.recv(timeout=1.0)
                except TimeoutError:
                    message = None
                if self._stopped:
                    logger.debug("stopped")
                    break
                if message is None:
                    sleep(0.1)
                    continue

                event_dict = json.loads(message)
                if "snapshot" in event_dict:
                    event_dict["snapshot"] =  Snapshot.from_nested_dict(event_dict["snapshot"])
                event_wrapper = EventWrapper(**event_dict)
                event = event_wrapper.event
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
