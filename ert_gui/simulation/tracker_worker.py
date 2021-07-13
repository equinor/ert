from ert_gui.model.snapshot import SnapshotModel
import typing
from ert_shared.status.tracker.evaluator import EvaluatorTracker
from ert_shared.status.tracker.legacy import LegacyTracker
from ert_shared.status.entity.event import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)
from qtpy.QtCore import QObject, Signal, Slot
import logging


logger = logging.getLogger(__name__)


class TrackerWorker(QObject):
    """A worker that consumes events produced by a tracker and emits them to qt
    subscribers."""

    consumed_event = Signal(object)
    done = Signal()

    def __init__(
        self, tracker: typing.Union[LegacyTracker, EvaluatorTracker], parent=None
    ):
        super().__init__(parent)
        logger.debug("init trackerworker")
        self._tracker = tracker
        self._stopped = False

    @Slot()
    def consume_and_emit(self):
        logger.debug("tracking...")
        for event in self._tracker.track():
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

        self._tracker.reset()
        self.done.emit()
        logger.debug("tracking done.")

    @Slot()
    def stop(self):
        logger.debug("stopping...")
        self._stopped = True

    @Slot()
    def request_termination(self):
        logger.debug("requesting termination...")
        self._tracker.request_termination()
        logger.debug("requested termination")
