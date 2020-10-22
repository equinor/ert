import asyncio
import copy
import json
import threading
import time

import websockets
from cloudevents.http import CloudEvent, to_json
from ert_shared.ensemble_evaluator.ws_util import wait_for_ws
from job_runner import JOBS_FILE
from res.job_queue import JobQueueManager
from res.job_queue.job_status_type_enum import JobStatusType

_FM_STAGE_WAITING = "com.equinor.ert.forward_model_stage.waiting"
_FM_STAGE_PENDING = "com.equinor.ert.forward_model_stage.pending"
_FM_STAGE_RUNNING = "com.equinor.ert.forward_model_stage.running"
_FM_STAGE_FAILURE = "com.equinor.ert.forward_model_stage.failure"
_FM_STAGE_SUCCESS = "com.equinor.ert.forward_model_stage.success"
_FM_STAGE_UNKNOWN = "com.equinor.ert.forward_model_stage.unknown"

_queue_state_to_event_type_map = {
    "JOB_QUEUE_NOT_ACTIVE": _FM_STAGE_UNKNOWN,
    "JOB_QUEUE_WAITING": _FM_STAGE_WAITING,
    "JOB_QUEUE_SUBMITTED": _FM_STAGE_PENDING,
    "JOB_QUEUE_PENDING": _FM_STAGE_PENDING,
    "JOB_QUEUE_RUNNING": _FM_STAGE_RUNNING,
    "JOB_QUEUE_DONE": _FM_STAGE_RUNNING,
    "JOB_QUEUE_EXIT": _FM_STAGE_RUNNING,
    "JOB_QUEUE_IS_KILLED": _FM_STAGE_RUNNING,
    "JOB_QUEUE_DO_KILL": _FM_STAGE_RUNNING,
    "JOB_QUEUE_SUCCESS": _FM_STAGE_SUCCESS,
    "JOB_QUEUE_RUNNING_DONE_CALLBACK": _FM_STAGE_RUNNING,
    "JOB_QUEUE_RUNNING_EXIT_CALLBACK": _FM_STAGE_RUNNING,
    "JOB_QUEUE_STATUS_FAILURE": _FM_STAGE_UNKNOWN,
    "JOB_QUEUE_FAILED": _FM_STAGE_FAILURE,
    "JOB_QUEUE_DO_KILL_NODE_FAILURE": _FM_STAGE_RUNNING,
    "JOB_QUEUE_UNKNOWN": _FM_STAGE_UNKNOWN,
}


def _queue_state_event_type(state):
    return _queue_state_to_event_type_map[state]


class JobQueueManagerAdaptor(JobQueueManager):

    # This adaptor is instantiated by code outside ERT's control, so for now
    # a class member is provided to allow this URL to vary
    ws_url = None

    ee_id = None

    def __init__(self, queue, queue_evaluators=None):
        super().__init__(queue, queue_evaluators)
        asyncio.set_event_loop(asyncio.new_event_loop())
        self._ws_url = self.ws_url
        self._changes_queue = asyncio.Queue()
        self._ee_id = self.ee_id
        wait_for_ws(self._ws_url)

        self._qindex_to_iens = {
            q_index: q_node.callback_arguments[0].iens
            for q_index, q_node in enumerate(self._queue.job_list)
        }
        self._state = [q_node.status.value for q_node in self._queue.job_list]

        self._patch_jobs_file()

        # JobQueueManager is at its core a while and a sleep. This is not
        # asyncio friendly, so create a separate thread from where websocket
        # events will be pushed.
        self._publisher_thread = threading.Thread(
            target=self._publisher, args=(asyncio.get_event_loop(),)
        )
        self._publisher_thread.start()

    def _patch_jobs_file(self):
        for q_index, q_node in enumerate(self._queue.job_list):
            with open(f"{q_node.run_path}/{JOBS_FILE}", "r+") as jobs_file:
                data = json.load(jobs_file)
                data["ee_id"] = self._ee_id
                data["real_id"] = self._qindex_to_iens[q_index]
                data["stage_id"] = 0
                jobs_file.seek(0)
                jobs_file.truncate()
                json.dump(data, jobs_file, indent=4)

    def _publisher(self, loop):
        loop.run_until_complete(
            self._publish_from_queue(self._changes_queue, self._ws_url)
        )

    @staticmethod
    def _translate_change_to_cloudevent(real_id, status):
        return CloudEvent(
            {
                "type": _queue_state_event_type(status),
                "source": f"/ert/ee/{0}/real/{real_id}/stage/{0}",
                "datacontenttype": "application/json",
            },
            {
                "queue_event_type": status,
            },
        )

    @staticmethod
    async def _publish_from_queue(queue, url):
        while True:
            async with websockets.connect(url) as websocket:
                changes = await queue.get()
                if changes is None:
                    await websocket.send("null")
                    return
                events = [
                    JobQueueManagerAdaptor._translate_change_to_cloudevent(
                        real_id, status
                    )
                    for real_id, status in changes.items()
                ]
                for event in events:
                    await websocket.send(to_json(event))

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
                    changes[self._qindex_to_iens[q_index]] = st
        return changes

    def _changes_after_transition(self):
        old_state, new_state = self._transition()
        return self._diff_states(old_state, new_state)

    def _snapshot(self):
        """Return the whole state"""
        snapshot = {}
        for q_index, state_val in enumerate(self._state):
            st = str(JobStatusType(state_val))
            snapshot[self._qindex_to_iens[q_index]] = st
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

        self._assert_complete()
        self._transition()
        self._publish_changes(self._snapshot())

        self._publish_changes(None)

        self._publisher_thread.join()
