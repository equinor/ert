import asyncio
import copy
import json
import threading
import time

import websockets
from ert_shared.ensemble_evaluator.entity import RealizationDecoder, create_realization
from ert_shared.ensemble_evaluator.ws_util import wait_for_ws
from res.job_queue import JobQueueManager
from res.job_queue.job_status_type_enum import JobStatusType


class JobQueueManagerAdaptor(JobQueueManager):

    # This adaptor is instantiated by code outside ERT's control, so for now
    # a class member is provided to allow this URL to vary
    ws_url = None

    def __init__(self, queue, queue_evaluators=None):
        super().__init__(queue, queue_evaluators)
        asyncio.set_event_loop(asyncio.new_event_loop())
        self._ws_url = self.ws_url
        self._changes_queue = asyncio.Queue()
        wait_for_ws(self._ws_url)

        self._qindex_to_iens = {
            q_index: q_node.callback_arguments[0].iens
            for q_index, q_node in enumerate(self._queue.job_list)
        }
        self._state = [q_node.status.value for q_node in self._queue.job_list]

        # JobQueueManager is at its core a while and a sleep. This is not
        # asyncio friendly, so create a separate thread from where websocket
        # events will be pushed.
        self._publisher_thread = threading.Thread(
            target=self._publisher, args=(asyncio.get_event_loop(),)
        )
        self._publisher_thread.start()

    def _publisher(self, loop):
        loop.run_until_complete(
            self._publish_from_queue(self._changes_queue, self._ws_url)
        )

    @staticmethod
    async def _publish_from_queue(queue, url):
        while True:
            async with websockets.connect(url) as websocket:
                changes = await queue.get()
                event = json.dumps(changes, cls=RealizationDecoder)
                await websocket.send(event)
                if changes is None:
                    return

    def _transition(self):
        """Transition to a new state, return both old and new state."""
        new_state = [job.status.value for job in self.queue.job_list]
        old_state = copy.copy(self._state)
        self._state = new_state
        return old_state, new_state

    def _diff_states(self, old_state, new_state):
        """Return the diff between old_state and new_state."""
        changes = {}

        diff = list(map(lambda s: s[0] == s[1], zip(old_state, new_state)))
        if len(diff) > 0:
            for q_index, equal in enumerate(diff):
                if not equal:
                    st = str(JobStatusType(new_state[q_index]))
                    changes[self._qindex_to_iens[q_index]] = create_realization(
                        st, None
                    )
        return changes

    def _changes_after_transition(self):
        old_state, new_state = self._transition()
        return self._diff_states(old_state, new_state)

    def _snapshot(self):
        """Return the whole state"""
        snapshot = {}
        for q_index, state_val in enumerate(self._state):
            st = str(JobStatusType(state_val))
            snapshot[self._qindex_to_iens[q_index]] = create_realization(st, None)
        return snapshot

    def _publish_changes(self, changes):
        """changes is a dict where each item is a iens to _Realization map.
        changes can be None, indicating EOT."""
        if changes is not None and len(changes) == 0:
            return

        asyncio.run_coroutine_threadsafe(
            self._changes_queue.put(changes), asyncio.get_event_loop()
        ).result()

    def execute_queue(self):
        self._publish_changes(self._snapshot())

        while self.queue.is_active() and not self.queue.stopped:
            self._launch_jobs()

            time.sleep(1)

            if self._queue_evaluators is not None:
                for func in self._queue_evaluators:
                    func()

            self._publish_changes(self._changes_after_transition())

        if self.queue.stopped:
            self._stop_jobs()

        self._transition()
        self._publish_changes(self._snapshot())

        self._assert_complete()

        self._publish_changes(None)

        self._publisher_thread.join()
