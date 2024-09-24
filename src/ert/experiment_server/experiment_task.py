import asyncio
import copy
import logging
import queue
from multiprocessing.queues import Queue
from typing import Dict, List, Optional

from fastapi.encoders import jsonable_encoder

from ert.config import QueueSystem
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.ensemble_evaluator.event import EndEvent, _UpdateEvent, FullSnapshotEvent, SnapshotUpdateEvent
from ert.run_models.base_run_model import BaseRunModel, StatusEvents


logger = logging.getLogger(__name__)


class EndTaskEvent:
    pass

class Subscriber:
    def __init__(self, compressed_events: bool) -> None:
        self.index = 0
        self.compressed_events = compressed_events
        self._event = asyncio.Event()

    def notify(self):
        self._event.set()

    async def wait_for_event(self):
        await self._event.wait()
        self._event.clear()

class ExperimentTask:
    def __init__(self, _id: str, model: BaseRunModel, status_queue: "Queue[StatusEvents]" ) -> None:
        self._id = _id
        self._model = model
        self.model_type = str(model.name())
        self._status_queue = status_queue
        self._subscribers: Dict[str, Subscriber] = {}
        self._events: List[StatusEvents] = []

        self._compressed_events: List[StatusEvents] = []
        self._allow_compression = False

    def cancel(self) -> None:
        if self._model is not None:
            self._model.cancel()

    async def run(self):
        loop = asyncio.get_running_loop()
        logger.info(f"Starting experiment {self._id}")

        port_range = None
        if self._model.queue_system == QueueSystem.LOCAL:
            port_range = range(49152, 51819)
        evaluator_server_config = EvaluatorServerConfig(custom_port_range=port_range)

        simulation_future = loop.run_in_executor(
            None,
            lambda: self._model.start_simulations_thread(
                evaluator_server_config
            ),
        )

        while True:
            try:
                item: StatusEvents = self._status_queue.get(block=False)
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue
            self._compressed_events.append(copy.deepcopy(item))

            if isinstance(item, _UpdateEvent):
                item.snapshot = item.snapshot.to_dict()
            event = jsonable_encoder(item)
            self._events.append(event)
            for sub in self._subscribers.values():
                sub.notify()
            await asyncio.sleep(0.1)

            if isinstance(item, EndEvent):
                self._events.append(EndTaskEvent())
                for sub in self._subscribers.values():
                    sub.notify()
                break

        await simulation_future
        self._compress_events()
        self._allow_compression = True
        self._model = None
        logger.info(f"Experiment {self._id} done")

    def _compress_events(self):
        items = self._compressed_events
        compressed_items = []
        for event in items:
            last_event = compressed_items[-1] if compressed_items else None
            if isinstance(last_event, FullSnapshotEvent) and isinstance(event, SnapshotUpdateEvent):
                last_event.snapshot.merge_snapshot(event.snapshot)
            elif isinstance(last_event, FullSnapshotEvent) and isinstance(event, FullSnapshotEvent):
                compressed_items.pop()
                compressed_items.append(event)
            else:
                compressed_items.append(event)

        self._compressed_events = []
        for item in compressed_items:
            if isinstance(item, _UpdateEvent):
                item.snapshot = item.snapshot.to_dict()
            event = jsonable_encoder(item)
            self._compressed_events.append(event)
        self._compressed_events.append(EndTaskEvent())


    async def get_event(self, subscriber_id: str) -> StatusEvents:
        if subscriber_id not in self._subscribers:
            self._subscribers[subscriber_id] = Subscriber(compressed_events=self._allow_compression)
        subscriber = self._subscribers[subscriber_id]

        event_list = self._compressed_events if subscriber.compressed_events else self._events

        while subscriber.index >= len(event_list):
            await subscriber.wait_for_event()

        event = event_list[subscriber.index]
        self._subscribers[subscriber_id].index += 1
        return event
