import asyncio
from collections import defaultdict, Mapping
from typing import Dict, List
import logging
from cloudevents.http import CloudEvent
from ert.ensemble_evaluator import identifiers as ids
from itertools import chain

flatten = chain.from_iterable

logger = logging.getLogger(__name__)


def nested_dict_keys(tree_structure: Dict) -> set:
    result = set()
    for key, value in tree_structure.items():
        result.add(key)
        if isinstance(value, Mapping):
            # TODO: raise if key exist.
            result.update(nested_dict_keys(value))
        else:
            result.update(set(value))
    return result


def merge_dict(left: Dict, right: Dict) -> None:
    for key, value in right.items():
        if isinstance(value, Mapping):
            merge_dict(left[key], right[key])
        else:
            left[key] = value


class StateMachine:
    def __init__(self, experiment_structure: Dict) -> None:
        self._ensemble_to_successful_realizations: Dict[int, List[int]] = defaultdict(
            list
        )
        self._entity_states: Dict[int, Dict] = defaultdict(
            dict, {k: {} for k in nested_dict_keys(experiment_structure)}
        )
        self._updated_entity_states: Dict[int, Dict] = defaultdict(dict)
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._experiment_structure = experiment_structure

        self._consume_task = asyncio.create_task(self.apply_updates())
        self._produce_task = asyncio.create_task(self.generate_transitions())
        self._update_lock = asyncio.Lock()
        self._update_interval = 0.5

    async def stop(self):
        self._consume_task.cancel()
        self._produce_task.cancel()

    def successful_realizations(self, iter_: int) -> int:
        """Return an integer indicating the number of successful realizations in an
        ensemble given iter_. Raise IndexError if the ensemble has no successful
        realizations."""
        return len(self._ensemble_to_successful_realizations[iter_])

    def add_successful_realization(self, iter_: int, real: int) -> None:
        logger.debug("adding successful real for iter %d, real: %d", iter_, real)
        if real not in self._ensemble_to_successful_realizations[iter_]:
            self._ensemble_to_successful_realizations[iter_].append(real)

    async def queue_event(self, event) -> None:
        """Adds the event to the queue to be processed later. Once the queue is processed the
        delta from previous processing of queue will be aggregated in updated_entity_states.
        In the scenario that the contents of a new event overlaps with a previous
        the old content will be overwritten.
        """
        if event["source"] not in self._entity_states.keys():
            raise KeyError(
                f"Provided source <{event['source']}> was not found in the experiment_structure"
            )
        await self._event_queue.put(event)

    async def get_update(self) -> Dict:
        """Returns the result of the last processing of the event queue (hence the delta from the last process).
        Does not include events currently in the queue. The content in get_update will be a subset of get_full_state
        and does not provide more information."""
        return self._updated_entity_states

    async def get_full_state(self) -> Dict:
        """Returns the state as it is described in `entity_states` (not including events in the queue)"""
        return self._entity_states

    async def apply_updates(self) -> None:
        self._updated_entity_states = defaultdict(dict)
        while True:
            event = await self._event_queue.get()
            self._event_queue.task_done()
            async with self._update_lock:
                merge_dict(
                    self._updated_entity_states[event["source"]],
                    {**event.data, "type": event["type"]},
                )

    async def generate_transitions(self):
        while True:
            await asyncio.sleep(self._update_interval)
            async with self._update_lock:
                merge_dict(self._entity_states, self._updated_entity_states)
                await self._generate_transitions()
                self._updated_entity_states = defaultdict(dict)

    async def _generate_transitions(self) -> None:
        return
        # if self._updated_entity_states:
        #      event = CloudEvent(
        #         {
        #             "type": ids.EVTYPE_FM_JOB_SUCCESS,
        #             "source": "ee/0/real/0/step/0/job/0",
        #         },
        #     )
        #     self._event_queue.put_nowait(event)
        # pass
