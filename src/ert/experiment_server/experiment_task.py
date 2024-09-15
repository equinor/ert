import asyncio
import queue
from multiprocessing.queues import Queue
from typing import Dict, List

from fastapi.encoders import jsonable_encoder

from ert.config import QueueSystem
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.ensemble_evaluator.event import EndEvent, _UpdateEvent
from ert.run_models.base_run_model import BaseRunModel, StatusEvents


class EndTaskEvent:
    pass

class Subscriber:
    def __init__(self) -> None:
        self.index = 0
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
        self._status_queue = status_queue
        self._subscribers: Dict[str, Subscriber] = {}
        self._events: List[StatusEvents] = []

    def cancel(self) -> None:
        self._model.cancel()

    async def run(self):
        loop = asyncio.get_running_loop()
        print(f"Starting experiment {self._id}")

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

            if isinstance(item, _UpdateEvent):
                item.snapshot = item.snapshot.to_dict()
            # print(item)
            # print()
            # print()
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
        print(f"Experiment {self._id} done")

    async def get_event(self, subscriber_id: str) -> StatusEvents:
        if subscriber_id not in self._subscribers:
            self._subscribers[subscriber_id] = Subscriber()
        subscriber = self._subscribers[subscriber_id]

        while subscriber.index >= len(self._events):
            await subscriber.wait_for_event()

        event = self._events[subscriber.index]
        self._subscribers[subscriber_id].index += 1
        return event
