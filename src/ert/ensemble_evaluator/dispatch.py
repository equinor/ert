import asyncio
import logging
import threading
import time
import traceback
from collections import OrderedDict, defaultdict
from typing import Callable, Mapping

from ert.ensemble_evaluator import identifiers

logger = logging.getLogger(__name__)


class BatchingDispatcher:  # pylint: disable=too-many-instance-attributes
    def __init__(self, sleep_between_batches_seconds, max_batch=1000):
        self._sleep_between_batches_seconds = sleep_between_batches_seconds
        self._max_batch = max_batch

        self._LOOKUP_MAP: Mapping[set, Callable] = defaultdict(lambda: (lambda _: None))
        self._running = True
        self._finished = False
        self._buffer = []
        self._buffer_lock = threading.Lock()

        self._dispatcher_loop = asyncio.new_event_loop()
        self._dispatcher_thread = threading.Thread(
            name="ert_ee_batch_dispatcher",
            target=self.run_dispatcher,
            args=(self._dispatcher_loop,),
        )
        self._dispatcher_thread.start()

    async def _work(self):
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
        function_to_events_map = OrderedDict()
        for f, event in batch_of_events_for_processing:
            if f not in function_to_events_map:
                function_to_events_map[f] = []
            function_to_events_map[f].append(event)

        def done_logger(_):
            logger.debug(
                f"processed {len(batch_of_events_for_processing)} events in "
                f"{(time.time()-t0):.6f}s. "
                f"{left_in_queue} left in queue"
            )

        events_handling = asyncio.gather(
            *[f(events) for f, events in function_to_events_map.items()]
        )
        events_handling.add_done_callback(done_logger)
        await events_handling

    def run_dispatcher(self, loop):
        try:
            loop.run_until_complete(self._job())
        except asyncio.CancelledError as ex:
            logger.warning(f"batcher was cancelled: {ex}")
        except Exception as failure:  # pylint: disable=broad-exception-caught
            logger.exception(f"exception in batching dispatcher: {failure}")
            trace_info = traceback.format_exception(
                type(failure), failure, failure.__traceback__
            )
            logger.error(f"{trace_info}")
        else:
            logger.debug("batcher finished normally")
            return
        finally:
            self._finished = True
        loop.run_until_complete(self._done_callback())
        logger.debug("Dispatcher thread exiting.")

    async def _done_callback(self):
        # call any registered handlers for FAILED. since we don't have
        # an event, pass empty list and let handler decide how to proceed
        done_func = self._LOOKUP_MAP[identifiers.EVTYPE_ENSEMBLE_FAILED]
        await done_func([])

    async def _job(self):
        while self._running:
            if len(self._buffer) < self._max_batch:
                time.sleep(self._sleep_between_batches_seconds)
            else:
                time.sleep(0)
            await self._work()
        # Make sure no events are lingering
        await self._work()

    async def join(self):
        self._running = False
        # if result is exception it should have been handled by
        # done-handler, but also avoid killing the caller here
        while not self._finished:
            await asyncio.sleep(0.01)

    def register_event_handler(self, event_types: set, function: Callable):
        for event_type in event_types:
            self._LOOKUP_MAP[event_type] = function

    async def handle_event(self, event):
        if not self._running:
            raise asyncio.InvalidStateError(
                "trying to handle event after batcher is done"
            )
        function = self._LOOKUP_MAP[event["type"]]
        with self._buffer_lock:
            self._buffer.append((function, event))
