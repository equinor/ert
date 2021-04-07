from collections import defaultdict, deque, OrderedDict

import asyncio
from typing import Optional

import ert_shared.ensemble_evaluator.entity.identifiers as identifiers
from ert_shared.status.entity.state import (
    ENSEMBLE_STATE_CANCELLED,
    ENSEMBLE_STATE_FAILED,
    ENSEMBLE_STATE_STOPPED,
)


class Batcher:
    def __init__(self, timeout, loop=None):
        self._timeout = timeout
        self._running = True
        self._buffer = deque()

        # Schedule task
        self._task = asyncio.ensure_future(self._job(), loop=loop)

    async def _work(self):
        event_buffer, self._buffer = self._buffer, deque()
        if event_buffer:
            function_to_events_map = OrderedDict()
            for f, event in event_buffer:
                if f not in function_to_events_map:
                    function_to_events_map[f] = []
                function_to_events_map[f].append(event)
            for f, events in function_to_events_map.items():
                await f(events)

    def put(self, f, event):
        self._buffer.append((f, event))

    async def _job(self):
        while self._running:
            await asyncio.sleep(self._timeout)
            await self._work()

        # Make sure no events are lingering
        await self._work()

    async def join(self):
        self._running = False
        await self._task


class Dispatcher:
    def __init__(self, ensemble, evaluator_callback, batcher=None):
        self._LOOKUP_MAP = defaultdict(list)
        self._batcher: Optional[Batcher] = batcher
        self._ensemble = ensemble
        self._evaluator_callback = evaluator_callback
        self._register_event_handlers()

    def register_event_handler(self, event_types, function, batching=False):
        if not isinstance(event_types, set):
            event_types = {event_types}
        for event_type in event_types:
            self._LOOKUP_MAP[event_type].append((function, batching))

    def _register_event_handlers(self):
        self.register_event_handler(
            event_types=identifiers.EVGROUP_FM_ALL,
            function=self._fm_handler,
            batching=True,
        )
        self.register_event_handler(
            event_types=identifiers.EVTYPE_ENSEMBLE_STOPPED,
            function=self._ensemble_stopped_handler,
            batching=True,
        )
        self.register_event_handler(
            event_types=identifiers.EVTYPE_ENSEMBLE_STARTED,
            function=self._ensemble_started_handler,
            batching=True,
        )
        self.register_event_handler(
            event_types=identifiers.EVTYPE_ENSEMBLE_CANCELLED,
            function=self._ensemble_cancelled_handler,
            batching=True,
        )
        self.register_event_handler(
            event_types=identifiers.EVTYPE_ENSEMBLE_FAILED,
            function=self._ensemble_failed_handler,
            batching=True,
        )

    async def _fm_handler(self, events):
        snapshot_update_event = self._ensemble.update_snapshot(events)
        await self._evaluator_callback(
            identifiers.EVTYPE_EE_SNAPSHOT_UPDATE, snapshot_update_event
        )

    async def _ensemble_stopped_handler(self, events):
        if self._ensemble.get_status() != ENSEMBLE_STATE_FAILED:
            await self._evaluator_callback(
                identifiers.EVTYPE_ENSEMBLE_STOPPED,
                self._ensemble.update_snapshot(events),
                events[0].data,
            )

    async def _ensemble_started_handler(self, events):
        if self._ensemble.get_status() != ENSEMBLE_STATE_FAILED:
            await self._evaluator_callback(
                identifiers.EVTYPE_ENSEMBLE_STARTED,
                self._ensemble.update_snapshot(events),
            )

    async def _ensemble_cancelled_handler(self, events):
        if self._ensemble.get_status() != ENSEMBLE_STATE_FAILED:
            await self._evaluator_callback(
                identifiers.EVTYPE_ENSEMBLE_CANCELLED,
                self._ensemble.update_snapshot(events),
            )

    async def _ensemble_failed_handler(self, events):
        if self._ensemble.get_status() not in [
            ENSEMBLE_STATE_STOPPED,
            ENSEMBLE_STATE_CANCELLED,
        ]:
            await self._evaluator_callback(
                identifiers.EVTYPE_ENSEMBLE_FAILED,
                self._ensemble.update_snapshot(events),
            )

    async def handle_event(self, event):
        for f, batching in self._LOOKUP_MAP[event["type"]]:
            if batching:
                if self._batcher is None:
                    raise RuntimeError(
                        f"Requested batching of {event} with handler {f}, but no batcher was specified"
                    )
                self._batcher.put(f, event)
            else:
                await f(event)
