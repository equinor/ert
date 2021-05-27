from collections import defaultdict, deque, OrderedDict

import asyncio
from typing import Optional


import logging
import threading
from contextlib import contextmanager
import pickle
from typing import Set
import cloudevents.exceptions
from http import HTTPStatus

import cloudpickle

import ert_shared.ensemble_evaluator.entity.identifiers as identifiers
import ert_shared.ensemble_evaluator.monitor as ee_monitor
import websockets
from websockets.legacy.server import WebSocketServerProtocol
from async_generator import asynccontextmanager
from cloudevents.http import from_json, to_json
from cloudevents.http.event import CloudEvent

from ert_shared.ensemble_evaluator.entity import serialization
from ert_shared.ensemble_evaluator.entity.snapshot import (
    PartialSnapshot,
    Snapshot,
    Job,
    Realization,
    SnapshotDict,
    Step,
)
from ert_shared.status.entity.state import (
    ENSEMBLE_STATE_CANCELLED,
    ENSEMBLE_STATE_FAILED,
    ENSEMBLE_STATE_STARTED,
    ENSEMBLE_STATE_STOPPED,
    JOB_STATE_START,
    REALIZATION_STATE_WAITING,
    STEP_STATE_UNKNOWN,
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
    def __init__(
        self, snapshot, ee_id, iter, clients, result_cb, stop_cb, batcher=None
    ):
        self._LOOKUP_MAP = defaultdict(list)
        self._batcher: Optional[Batcher] = batcher
        self._snapshot = snapshot
        self._ee_id = ee_id
        self._iter = iter
        self._clients = clients

        self._result_cb = result_cb
        self._stop_cb = stop_cb

        self._register_event_handlers()

    def register_event_handler(self, event_types, batching=False):
        def decorator(function):
            nonlocal event_types, batching
            if not isinstance(event_types, set):
                event_types = set({event_types})
            for event_type in event_types:
                self._LOOKUP_MAP[event_type].append((function, batching))

            def wrapper(*args, **kwargs):
                return function(*args, **kwargs)

            return wrapper

        return decorator

    def _register_event_handlers(self):
        self.register_event_handler(identifiers.EVGROUP_FM_ALL, batching=True)(
            self._fm_handler
        )
        self.register_event_handler(identifiers.EVTYPE_ENSEMBLE_STOPPED, batching=True)(
            self._ensemble_stopped_handler
        )
        self.register_event_handler(identifiers.EVTYPE_ENSEMBLE_STARTED, batching=True)(
            self._ensemble_started_handler
        )
        self.register_event_handler(
            identifiers.EVTYPE_ENSEMBLE_CANCELLED, batching=True
        )(self._ensemble_cancelled_handler)
        self.register_event_handler(identifiers.EVTYPE_ENSEMBLE_FAILED, batching=True)(
            self._ensemble_failed_handler
        )

    async def _fm_handler(self, events):
        snapshot_mutate_event = PartialSnapshot(self._snapshot)
        for event in events:
            snapshot_mutate_event.from_cloudevent(event)
        await self._send_snapshot_update(snapshot_mutate_event)

    async def _ensemble_stopped_handler(self, events):
        for event in events:
            # self._result = event.data
            self._result_cb(event.data)
            if self._snapshot.get_status() != ENSEMBLE_STATE_FAILED:
                snapshot_mutate_event = PartialSnapshot(self._snapshot).from_cloudevent(
                    event
                )
                await self._send_snapshot_update(snapshot_mutate_event)

    async def _ensemble_started_handler(self, events):
        for event in events:
            if self._snapshot.get_status() != ENSEMBLE_STATE_FAILED:
                snapshot_mutate_event = PartialSnapshot(self._snapshot).from_cloudevent(
                    event
                )
                await self._send_snapshot_update(snapshot_mutate_event)

    async def _ensemble_cancelled_handler(self, events):
        for event in events:
            if self._snapshot.get_status() != ENSEMBLE_STATE_FAILED:
                snapshot_mutate_event = PartialSnapshot(self._snapshot).from_cloudevent(
                    event
                )
                await self._send_snapshot_update(snapshot_mutate_event)
                self._stop_cb()

    async def _ensemble_failed_handler(self, events):
        for event in events:
            if self._snapshot.get_status() not in [
                ENSEMBLE_STATE_STOPPED,
                ENSEMBLE_STATE_CANCELLED,
            ]:
                snapshot_mutate_event = PartialSnapshot(self._snapshot).from_cloudevent(
                    event
                )
                await self._send_snapshot_update(snapshot_mutate_event)

    async def _send_snapshot_update(self, snapshot_mutate_event):
        self._snapshot.merge_event(snapshot_mutate_event)
        out_cloudevent = CloudEvent(
            {
                "type": identifiers.EVTYPE_EE_SNAPSHOT_UPDATE,
                "source": f"/ert/ee/{self._ee_id}",
            },
            snapshot_mutate_event.to_dict(),
        )
        out_cloudevent.data["iter"] = self._iter
        out_msg = to_json(
            out_cloudevent, data_marshaller=serialization.evaluator_marshaller
        ).decode()

        if out_msg and self._clients:
            await asyncio.wait([client.send(out_msg) for client in self._clients])

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
