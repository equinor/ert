import logging
from queue import SimpleQueue

from qtpy.QtCore import QObject, Signal, Slot

from ert.ensemble_evaluator import EndEvent, FullSnapshotEvent, SnapshotUpdateEvent
from ert.gui.model.snapshot import SnapshotModel

logger = logging.getLogger(__name__)


class QueueEmitter(QObject):
    """A worker that emits items put on a queue to qt subscribers."""

    new_event = Signal(object)
    done = Signal()

    def __init__(
        self,
        event_queue: SimpleQueue,
        parent=None,
    ):
        super().__init__(parent)
        logger.debug("init QueueEmitter")
        self._event_queue = event_queue
        self._stopped = False

    @Slot()
    def consume_and_emit(self):
        logger.debug("tracking...")
        while True:
            event = self._event_queue.get()
            if self._stopped:
                logger.debug("stopped")
                break

            # pre-rendering in this thread to avoid work in main rendering thread
            if isinstance(event, FullSnapshotEvent) and event.snapshot:
                SnapshotModel.prerender(event.snapshot)
            elif isinstance(event, SnapshotUpdateEvent) and event.partial_snapshot:
                SnapshotModel.prerender(event.partial_snapshot)

            logger.debug(f"emit {event}")
            self.new_event.emit(event)

            if isinstance(event, EndEvent):
                logger.debug("got end event")
                break

        self.done.emit()
        logger.debug("tracking done.")

    @Slot()
    def stop(self):
        logger.debug("stopping...")
        self._stopped = True
