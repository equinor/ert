import logging
from typing import Callable, Iterator, Union

from qtpy.QtCore import QObject, Signal, Slot

from ert.ensemble_evaluator import EndEvent, FullSnapshotEvent, SnapshotUpdateEvent
from ert.gui.model.snapshot import SnapshotModel

logger = logging.getLogger(__name__)


class TrackerWorker(QObject):
    """A worker that consumes events produced by a tracker and emits them to qt
    subscribers."""

    consumed_event = Signal(object)
    done = Signal()

    def __init__(
        self,
        event_generator_factory: Callable[
            [], Iterator[Union[FullSnapshotEvent, SnapshotUpdateEvent, EndEvent]]
        ],
        parent=None,
    ):
        super().__init__(parent)
        logger.debug("init trackerworker")
        self._tracker = event_generator_factory
        self._stopped = False

    @Slot()
    def consume_and_emit(self):
        logger.debug("tracking...")
        for event in self._tracker():
            if self._stopped:
                logger.debug("stopped")
                break

            if isinstance(event, FullSnapshotEvent) and event.snapshot:
                SnapshotModel.prerender(event.snapshot)
            elif isinstance(event, SnapshotUpdateEvent) and event.partial_snapshot:
                SnapshotModel.prerender(event.partial_snapshot)

            logger.debug(f"emit {event}")
            self.consumed_event.emit(event)

            if isinstance(event, EndEvent):
                logger.debug("got end event")
                break

        self.done.emit()
        logger.debug("tracking done.")

    @Slot()
    def stop(self):
        logger.debug("stopping...")
        self._stopped = True
