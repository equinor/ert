import logging
from queue import Empty
import threading
import time
import queue

import dateutil.parser
import ert_shared.ensemble_evaluator.entity.identifiers as ids
from ert_shared.ensemble_evaluator.entity.tool import recursive_update
from ert_shared.ensemble_evaluator.monitor import create as create_ee_monitor
from ert_shared.tracker.events import DetailedEvent, GeneralEvent
from res.job_queue import ForwardModelJobStatus, JobStatusType
import copy


class EvaluatorTracker:
    def __init__(self, model, ee_monitor_connection_details, states):
        self._model = model
        self._states = states

        host, port = ee_monitor_connection_details

        # The state mutex guards all code paths that access the _realization_progress state. It also guards the _updates
        # list since the same code paths access that
        self._state_mutex = threading.Lock()
        # The _realization_progress is all the state needed to produce the UI events
        self._realization_progress = {}
        # The _updates contain the snapshot update events. The event generators will read this and update the
        # _realization_progress before producing a new event.
        self._updates = []

        # For every received event (except terminated), an item is put on the queue.
        # Each time a GeneralEvent is produced, all items on the queue are
        # marked as done. This let's the event loop halt until all messages are processed before exiting
        self._work_queue = queue.Queue()

        self._drainer_thread = threading.Thread(
            target=self._drain_monitor, args=(host, port)
        )
        self._drainer_thread.start()

    def _drain_monitor(self, host, port):
        drainer_logger = logging.getLogger("ert_shared.ensemble_evaluator.drainer")
        monitor = create_ee_monitor(host, port)
        while monitor:
            try:
                for event in monitor.track():
                    if event["type"] == ids.EVTYPE_EE_SNAPSHOT:
                        iter_ = event.data["metadata"]["iter"]
                        with self._state_mutex:
                            self._realization_progress[
                                iter_
                            ] = self._snapshot_to_realization_progress(event.data)
                            self._work_queue.put(None)
                    elif event["type"] == ids.EVTYPE_EE_SNAPSHOT_UPDATE:
                        with self._state_mutex:
                            self._updates.append(event.data)
                            self._work_queue.put(None)
                    elif event["type"] == ids.EVTYPE_EE_TERMINATED:
                        drainer_logger.debug("got terminator event")
                        while True:
                            if self._model.isFinished():
                                drainer_logger.debug(
                                    "observed that model was finished, waiting tasks completion..."
                                )
                                self._work_queue.join()
                                drainer_logger.debug("tasks complete")
                                return
                            try:
                                time.sleep(5)
                                drainer_logger.debug("connecting to new monitor...")
                                monitor = create_ee_monitor(host, port)
                                drainer_logger.debug("connected")
                                break
                            except ConnectionRefusedError as e:
                                drainer_logger.debug(f"connection refused: {e}")
                                pass

            except ConnectionRefusedError as e:
                if self._model.isFinished():
                    return
                else:
                    raise e

    def _get_most_recent_snapshot(self):
        iter_ = self._get_current_iter()
        if iter_ is None:
            return None
        return self._realization_progress[iter_]

    def _get_current_iter(self):
        """Return None if there's no current iteration. Means we've never
        received a snapshot."""
        if len(self._realization_progress) == 0:
            return None
        return max([int(key) for key in self._realization_progress.keys()])

    def _count_in_state(self, state):
        snapshot = self._get_most_recent_snapshot()
        if snapshot is None:
            return 0

        count = 0
        for job_status_tuple in snapshot.values():
            step_status = job_status_tuple[1][1]
            if step_status.lower == "finished":
                print("finished")
            if step_status.lower() == state.lower():
                count += 1
        return count

    def _get_ensemble_size(self):
        snapshot = self._get_most_recent_snapshot()
        if snapshot is None:
            return 0
        return len(snapshot)

    def is_finished(self):
        return not self._drainer_thread.is_alive()

    def general_event(self):
        phase_name = self._model.getPhaseName()
        phase = self._model.currentPhase()
        phase_count = self._model.phaseCount()

        done_count = 0
        with self._state_mutex:
            for update in self._updates:
                self._update_states(update)
            self._updates.clear()

            total_size = self._get_ensemble_size()
            for state in self._states:
                state.count = self._count_in_state(state.name)
                state.total_count = total_size

                if state.name == "Finished":
                    done_count = state.count

            try:
                while True:
                    self._work_queue.get_nowait()
                    self._work_queue.task_done()
            except queue.Empty:
                pass

        try:
            progress = float(done_count) / total_size
        except ZeroDivisionError:
            progress = 0

        return GeneralEvent(
            phase_name,
            phase,
            phase_count,
            progress,
            self._model.isIndeterminate(),
            self._states,
            self._model.get_runtime(),
        )

    def _snapshot_to_realization_progress(self, snapshot):
        realization_progress = {}

        for iens, real in snapshot["reals"].items():
            iens_int = int(iens)
            queue_state_str = real["stages"]["0"].get("queue_state")
            queue_state = (
                None
                if queue_state_str is None
                else JobStatusType.from_string(queue_state_str)
            )
            realization_progress[iens_int] = (
                [
                    ForwardModelJobStatus(
                        job["name"],
                        status=job["status"],
                        start_time=dateutil.parser.parse(job["start_time"])
                        if job["start_time"] is not None
                        else None,
                        end_time=dateutil.parser.parse(job["end_time"])
                        if job["end_time"] is not None
                        else None,
                        current_memory_usage=job.get("data", {}).get(
                            "current_memory_usage"
                        ),
                        max_memory_usage=job.get("data", {}).get("max_memory_usage"),
                    )
                    for job in real["stages"]["0"]["steps"]["0"]["jobs"].values()
                ],
                (queue_state, real["stages"]["0"]["steps"]["0"]["status"]),
            )
        return realization_progress

    def _update_states(self, snapshot_updates):

        realization_progress = self._get_most_recent_snapshot()
        if "reals" in snapshot_updates:
            for iens, real in snapshot_updates["reals"].items():
                iens_int = int(iens)
                job_statuses = realization_progress[iens_int][0]
                queue_state = realization_progress[iens_int][1][0]
                step_state = realization_progress[iens_int][1][1]

                if (
                    real.get("stages", {})
                    .get("0", {})
                    .get("steps", {})
                    .get("0", {})
                    .get("status")
                ):
                    step_state = real["stages"]["0"]["steps"]["0"]["status"]

                if real.get("stages", {}).get("0", {}).get("queue_state"):
                    queue_state_str = real["stages"]["0"].get("queue_state")
                    realization_progress[iens_int] = (
                        job_statuses,
                        JobStatusType.from_string(queue_state_str),
                    )

                if "jobs" in real.get("stages", {}).get("0", {}).get("steps", {}).get(
                    "0", {}
                ):
                    for job_id, job in real["stages"]["0"]["steps"]["0"][
                        "jobs"
                    ].items():
                        job_id = int(job_id)
                        status = realization_progress[iens_int][0][job_id]
                        if job.get("status"):
                            status.status = job["status"]
                        if job.get("start_time"):
                            status.start_time = dateutil.parser.parse(job["start_time"])
                        if job.get("end_time"):
                            status.end_time = dateutil.parser.parse(job["end_time"])
                        if job.get("data", {}).get("current_memory_usage"):
                            status.current_memory_usage = job["data"][
                                "current_memory_usage"
                            ]
                        if job.get("data", {}).get("max_memory_usage"):
                            status.max_memory_usage = job["data"]["max_memory_usage"]

                realization_progress[iens_int] = (
                    job_statuses,
                    (queue_state, step_state),
                )

    def detailed_event(self):
        with self._state_mutex:
            for update in self._updates:
                self._update_states(update)
            self._updates.clear()
            iter_ = self._get_current_iter()
            if iter_ is None:
                # the gui seems to handle negative iter
                iter_ = -1
                realization_progress = {}
            else:
                realization_progress = self._realization_progress
            return DetailedEvent(realization_progress, iter_)
