import asyncio
import logging
import threading
import time
import traceback
from collections import OrderedDict, defaultdict
from typing import Callable, Collection, Dict, List, Tuple

from cloudevents.http import CloudEvent

from _ert.threading import ErtThread
from ert.ensemble_evaluator import identifiers

logger = logging.getLogger(__name__)


class BatchingDispatcher:
    def __init__(
        self, sleep_between_batches_seconds: int, max_batch: int = 1000
    ) -> None:
        self._sleep_between_batches_seconds = sleep_between_batches_seconds
        self._max_batch = max_batch

        def log_unknown_event_type(events: List[CloudEvent]) -> None:
            if len(events) > 0:
                event_type = events[0]["type"]
                logger.warning(
                    "tried to lookup handle function for unknown "
                    f"event type: {event_type}"
                )

        self._LOOKUP_MAP: Dict[str, Callable[[List[CloudEvent]], None]] = defaultdict(
            lambda: log_unknown_event_type
        )
        self._running = True
        self._finished = False
        self._buffer: List[Tuple[Callable[[List[CloudEvent]], None], CloudEvent]] = []
        self._buffer_lock = threading.Lock()

        self._dispatcher_thread = ErtThread(
            name="ert_ee_batch_dispatcher",
            target=self.run_dispatcher,
        )
        self._dispatcher_thread.start()

    def _work(self) -> None:
        if len(self._buffer) == 0:
            logger.debug("no events to be processed in queue")
            return

        with self._buffer_lock:
            t0 = time.time()
            batch_of_events_for_processing, self._buffer = (
                self._buffer[: self._max_batch],
                self._buffer[self._max_batch :],
            )
            left_in_queue = len(self._buffer)
        function_to_events_map: Dict[
            Callable[[List[CloudEvent]], None], List[CloudEvent]
        ] = OrderedDict()
        for func, event in batch_of_events_for_processing:
            if func not in function_to_events_map:
                function_to_events_map[func] = []
            function_to_events_map[func].append(event)

        for func, events in function_to_events_map.items():
            func(events)

        logger.debug(
            f"processed {len(batch_of_events_for_processing)} events in "
            f"{(time.time()-t0):.6f}s. "
            f"{left_in_queue} left in queue"
        )

    def run_dispatcher(self) -> None:
        try:
            self._job()
        except Exception as failure:
            logger.exception(f"exception in batching dispatcher: {failure}")
            trace_info = traceback.format_exception(
                type(failure), failure, failure.__traceback__
            )
            logger.exception(f"{trace_info}")
        else:
            logger.debug("batcher finished normally")
            return
        finally:
            self._finished = True

        # call any registered handlers for FAILED. since we don't have
        # an event, pass empty list and let handler decide how to proceed
        failure_func = self._LOOKUP_MAP[identifiers.EVTYPE_ENSEMBLE_FAILED]
        failure_func([])

        logger.debug("Dispatcher thread exiting.")

    def _job(self) -> None:
        while self._running:
            if len(self._buffer) < self._max_batch:
                time.sleep(self._sleep_between_batches_seconds)
            else:
                # sleep(0) is used to play nice with other threads.
                # It will release the GIL and let other threads run.
                time.sleep(0)
            self._work()
        # Make sure no events are lingering
        self._work()

    async def wait_until_finished(self) -> None:
        self._running = False
        while not self._finished:
            await asyncio.sleep(0.01)

    def set_event_handler(
        self, event_types: Collection[str], function: Callable[[List[CloudEvent]], None]
    ) -> None:
        for event_type in event_types:
            self._LOOKUP_MAP[event_type] = function

    async def handle_event(self, event: CloudEvent) -> None:
        if not self._running:
            raise asyncio.InvalidStateError(
                "trying to handle event after batcher is done"
            )
        function = self._LOOKUP_MAP[event["type"]]
        with self._buffer_lock:
            self._buffer.append((function, event))
