import asyncio
from ert_shared.status.queue_diff import QueueDiff
import json
import logging

import websockets
from cloudevents.http import CloudEvent, to_json
from job_runner import JOBS_FILE

logger = logging.getLogger(__name__)

_FM_STAGE_WAITING = "com.equinor.ert.forward_model_stage.waiting"
_FM_STAGE_PENDING = "com.equinor.ert.forward_model_stage.pending"
_FM_STAGE_RUNNING = "com.equinor.ert.forward_model_stage.running"
_FM_STAGE_FAILURE = "com.equinor.ert.forward_model_stage.failure"
_FM_STAGE_SUCCESS = "com.equinor.ert.forward_model_stage.success"
_FM_STAGE_UNKNOWN = "com.equinor.ert.forward_model_stage.unknown"

_queue_state_to_event_type_map = {
    "JOB_QUEUE_NOT_ACTIVE": _FM_STAGE_WAITING,
    "JOB_QUEUE_WAITING": _FM_STAGE_WAITING,
    "JOB_QUEUE_SUBMITTED": _FM_STAGE_WAITING,
    "JOB_QUEUE_PENDING": _FM_STAGE_PENDING,
    "JOB_QUEUE_RUNNING": _FM_STAGE_RUNNING,
    "JOB_QUEUE_DONE": _FM_STAGE_RUNNING,
    "JOB_QUEUE_EXIT": _FM_STAGE_RUNNING,
    "JOB_QUEUE_IS_KILLED": _FM_STAGE_FAILURE,
    "JOB_QUEUE_DO_KILL": _FM_STAGE_FAILURE,
    "JOB_QUEUE_SUCCESS": _FM_STAGE_SUCCESS,
    "JOB_QUEUE_RUNNING_DONE_CALLBACK": _FM_STAGE_RUNNING,
    "JOB_QUEUE_RUNNING_EXIT_CALLBACK": _FM_STAGE_RUNNING,
    "JOB_QUEUE_STATUS_FAILURE": _FM_STAGE_UNKNOWN,
    "JOB_QUEUE_FAILED": _FM_STAGE_FAILURE,
    "JOB_QUEUE_DO_KILL_NODE_FAILURE": _FM_STAGE_FAILURE,
    "JOB_QUEUE_UNKNOWN": _FM_STAGE_UNKNOWN,
}


def _queue_state_event_type(state):
    return _queue_state_to_event_type_map[state]


class QueueAdaptor:
    def __init__(self, queue, config, ee_id):
        self._ws_url = config.get("dispatch_url")
        self._ee_id = ee_id
        self._queue = queue
        self._diff = QueueDiff(queue)

        self._patch_jobs_file()

    def _patch_jobs_file(self):
        for q_index, q_node in enumerate(self._queue.job_list):
            with open(f"{q_node.run_path}/{JOBS_FILE}", "r+") as jobs_file:
                data = json.load(jobs_file)
                data["ee_id"] = self._ee_id
                data["real_id"] = self._diff.iens_from_queue_index(q_index)
                data["stage_id"] = 0
                jobs_file.seek(0)
                jobs_file.truncate()
                json.dump(data, jobs_file, indent=4)

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
    async def _publish_changes(changes, websocket):
        events = [
            QueueAdaptor._translate_change_to_cloudevent(real_id, status)
            for real_id, status in changes.items()
        ]
        for event in events:
            await websocket.send(to_json(event))

    async def execute_queue(self, pool_sema, evaluators):
        async with websockets.connect(self._ws_url) as websocket:
            await self._publish_changes(self._diff.snapshot(), websocket)

            try:
                while self._queue.is_active() and not self._queue.stopped:
                    self._queue.launch_jobs(pool_sema)

                    await asyncio.sleep(1)

                    if evaluators is not None:
                        for func in evaluators:
                            func()

                    await self._publish_changes(
                        self._diff.changes_after_transition(), websocket
                    )
            except asyncio.CancelledError:
                if self._queue.stopped:
                    logger.debug(
                        "observed that the queue had stopped after cancellation, stopping jobs..."
                    )
                    self._queue.stop_jobs()
                    logger.debug("jobs now stopped (after cancellation)")
                raise

            if self._queue.stopped:
                logger.debug("observed that the queue had stopped, stopping jobs...")
                await self._queue.stop_jobs_async()
                logger.debug("jobs now stopped")
            self._queue.assert_complete()
            self._diff.transition()
            await self._publish_changes(self._diff.snapshot(), websocket)
