from ert_shared.status.utils import tracker_progress
from ert_shared.status.entity.state import (
    ENSEMBLE_STATE_CANCELLED,
    ENSEMBLE_STATE_STOPPED,
    ENSEMBLE_STATE_FAILED,
)
import logging
import queue
import threading
import time
from ert_shared.models.base_run_model import BaseRunModel
import ert_shared.ensemble_evaluator.entity.identifiers as ids
from ert_shared.ensemble_evaluator.entity.snapshot import PartialSnapshot, Snapshot
from ert_shared.ensemble_evaluator.monitor import create as create_ee_monitor
from ert_shared.ensemble_evaluator.ws_util import wait_for_ws
from ert_shared.status.entity.event import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)


class OutOfOrderSnapshotUpdateException(ValueError):
    pass


class EvaluatorTracker:
    def __init__(
        self,
        model: BaseRunModel,
        host,
        port,
        general_interval,
        detailed_interval,
    ):
        self._model = model

        self._monitor_host = host
        self._monitor_port = port
        self._monitor_url = f"ws://{host}:{port}"

        self._work_queue = queue.Queue()

        self._drainer_thread = threading.Thread(
            target=self._drain_monitor, name="DrainerThread"
        )
        self._drainer_thread.start()

        self._iter_snapshot = {}

    def _drain_monitor(self):
        drainer_logger = logging.getLogger("ert_shared.ensemble_evaluator.drainer")
        failures = 0
        while not self._model.isFinished():
            try:
                drainer_logger.debug("connecting to new monitor...")
                with create_ee_monitor(
                    self._monitor_host, self._monitor_port
                ) as monitor:
                    drainer_logger.debug("connected")
                    for event in monitor.track():
                        if event["type"] in (
                            ids.EVTYPE_EE_SNAPSHOT,
                            ids.EVTYPE_EE_SNAPSHOT_UPDATE,
                        ):
                            self._work_queue.put(event)
                            if event.data.get(ids.STATUS) in [
                                ENSEMBLE_STATE_STOPPED,
                                ENSEMBLE_STATE_FAILED,
                            ]:
                                drainer_logger.debug(
                                    "observed evaluation stopped event, signal done"
                                )
                                monitor.signal_done()
                            if event.data.get(ids.STATUS) == ENSEMBLE_STATE_CANCELLED:
                                drainer_logger.debug(
                                    "observed evaluation cancelled event, exit drainer"
                                )
                                return
                        elif event["type"] == ids.EVTYPE_EE_TERMINATED:
                            drainer_logger.debug("got terminator event")

                # This sleep needs to be there. Refer to issue #1250: `Authority
                # on information about evaluations/experiments`
                time.sleep(5)

            except ConnectionRefusedError as e:
                if not self._model.isFinished():
                    drainer_logger.debug(f"connection refused: {e}")
                    failures += 1
                    if failures == 10:
                        drainer_logger.debug(f"giving up.")
                        raise e
            else:
                failures = 0

        drainer_logger.debug(
            "observed that model was finished, waiting tasks completion..."
        )
        # The model has finished, we indicate this by sending a None
        self._work_queue.put(None)
        self._work_queue.join()
        drainer_logger.debug("tasks complete")

    def track(self):
        while True:
            event = self._work_queue.get()
            if event is None:
                try:
                    yield EndEvent(
                        failed=self._model.hasRunFailed(),
                        failed_msg=self._model.getFailMessage(),
                    )
                except GeneratorExit:
                    # consumers may exit at this point, make sure the last
                    # task is marked as done
                    self._work_queue.task_done()
                break
            elif event["type"] == ids.EVTYPE_EE_SNAPSHOT:
                iter_ = event.data["iter"]
                snapshot = Snapshot(event.data)
                self._iter_snapshot[iter_] = snapshot
                yield FullSnapshotEvent(
                    phase_name=self._model.getPhaseName(),
                    current_phase=self._model.currentPhase(),
                    total_phases=self._model.phaseCount(),
                    indeterminate=self._model.isIndeterminate(),
                    progress=self._progress(),
                    iteration=iter_,
                    snapshot=snapshot,
                )
            elif event["type"] == ids.EVTYPE_EE_SNAPSHOT_UPDATE:
                iter_ = event.data["iter"]
                if iter_ not in self._iter_snapshot:
                    raise OutOfOrderSnapshotUpdateException(
                        f"got {ids.EVTYPE_EE_SNAPSHOT_UPDATE} without having stored snapshot for iter {iter_}"
                    )
                partial = PartialSnapshot(self._iter_snapshot[iter_]).from_cloudevent(
                    event
                )
                yield SnapshotUpdateEvent(
                    phase_name=self._model.getPhaseName(),
                    current_phase=self._model.currentPhase(),
                    total_phases=self._model.phaseCount(),
                    indeterminate=self._model.isIndeterminate(),
                    progress=self._progress(),
                    iteration=iter_,
                    partial_snapshot=partial,
                )

            self._work_queue.task_done()

    def is_finished(self):
        return not self._drainer_thread.is_alive()

    def _progress(self) -> float:
        return tracker_progress(self)

    def reset(self):
        pass

    def _clear_work_queue(self):
        try:
            while True:
                self._work_queue.get_nowait()
                self._work_queue.task_done()
        except queue.Empty:
            pass

    def request_termination(self):
        logger = logging.getLogger("ert_shared.ensemble_evaluator.tracker")

        # There might be some situations where the
        # evaluation is finished or the evaluation
        # is yet to start when calling this function.
        # In these cases the monitor is not started
        #
        # To avoid waiting too long we exit if we are not
        # able to connect to the monitor after 2 tries
        #
        # See issue: https://github.com/equinor/ert/issues/1250
        #
        try:
            wait_for_ws(self._monitor_url, 2)
        except ConnectionRefusedError as e:
            logger.warning(f"{__name__} - exception {e}")
            return

        with create_ee_monitor(self._monitor_host, self._monitor_port) as monitor:
            monitor.signal_cancel()
        while self._drainer_thread.is_alive():
            self._clear_work_queue()
            time.sleep(1)
