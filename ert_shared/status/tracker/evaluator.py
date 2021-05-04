from cloudevents.http.event import CloudEvent
from datetime import datetime
from typing import List, Optional, Union
from ert_shared.status.utils import tracker_progress
from ert_shared.status.entity.state import (
    ENSEMBLE_STATE_CANCELLED,
    ENSEMBLE_STATE_STOPPED,
    ENSEMBLE_STATE_FAILED,
)
import itertools
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
    DONE = None

    def __init__(
        self,
        model: BaseRunModel,
        host,
        port,
        general_interval,
        detailed_interval,
        token=None,
        cert=None,
        next_ensemble_evaluator_wait_time=5,
    ):
        self._model = model

        self._monitor_host = host
        self._monitor_port = port
        self._token = token
        self._cert = cert
        self._protocol = "ws" if cert is None else "wss"
        self._monitor_url = f"{self._protocol}://{host}:{port}"
        self._next_ensemble_evaluator_wait_time = next_ensemble_evaluator_wait_time

        self._work_queue = queue.Queue()

        self._drainer_thread = threading.Thread(
            target=self._drain_monitor,
            name="DrainerThread",
        )
        self._drainer_thread.start()

        self._iter_snapshot = {}
        self._general_interval = general_interval

    def _drain_monitor(self):
        drainer_logger = logging.getLogger("ert_shared.ensemble_evaluator.drainer")
        failures = 0
        while not self._model.isFinished():
            try:
                drainer_logger.debug("connecting to new monitor...")
                with create_ee_monitor(
                    self._monitor_host,
                    self._monitor_port,
                    protocol=self._protocol,
                    cert=self._cert,
                    token=self._token,
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
                                # Allow track() to emit an EndEvent.
                                self._work_queue.put(EvaluatorTracker.DONE)
                                return
                        elif event["type"] == ids.EVTYPE_EE_TERMINATED:
                            drainer_logger.debug("got terminator event")

                # This sleep needs to be there. Refer to issue #1250: `Authority
                # on information about evaluations/experiments`
                time.sleep(self._next_ensemble_evaluator_wait_time)

            except ConnectionRefusedError as e:
                if not self._model.isFinished():
                    drainer_logger.debug(f"connection refused: {e}")
                    failures += 1
                    if failures == 10:
                        drainer_logger.debug("giving up.")
                        raise e
            else:
                failures = 0

        drainer_logger.debug(
            "observed that model was finished, waiting tasks completion..."
        )
        # The model has finished, we indicate this by sending a None
        self._work_queue.put(EvaluatorTracker.DONE)
        self._work_queue.join()
        drainer_logger.debug("tasks complete")

    def track(self):
        done: bool = False
        while not done:
            events: List[CloudEvent] = []

            while True:
                try:
                    event: CloudEvent = self._work_queue.get_nowait()
                    if event is EvaluatorTracker.DONE:
                        done = True
                        break
                    events.append(event)
                except queue.Empty:
                    break

            if events:
                yield from self._batch(events)
            time.sleep(self._general_interval)

        try:
            yield EndEvent(
                failed=self._model.hasRunFailed(),
                failed_msg=self._model.getFailMessage(),
            )
        except GeneratorExit:
            # consumers may exit at this point, make sure the last
            # task is marked as done
            pass
        self._work_queue.task_done()

    def _flush(self, batch: List[CloudEvent]) -> SnapshotUpdateEvent:
        iter_: int = batch[0].data["iter"]
        partial: PartialSnapshot = PartialSnapshot(self._iter_snapshot[iter_])
        for event in batch:
            partial.from_cloudevent(event)
        self._iter_snapshot[iter_].merge_event(partial)
        update_event = SnapshotUpdateEvent(
            phase_name=self._model.getPhaseName(),
            current_phase=self._model.currentPhase(),
            total_phases=self._model.phaseCount(),
            indeterminate=self._model.isIndeterminate(),
            progress=self._progress(),
            iteration=iter_,
            partial_snapshot=partial,
        )
        for _ in range(len(batch)):
            self._work_queue.task_done()
        return update_event

    def _batch(self, events):
        batch: List[CloudEvent] = []

        for event in events:
            if event["type"] == ids.EVTYPE_EE_SNAPSHOT:

                # A new iteration, so ensure any updates for the previous one,
                # is emitted.
                if batch:
                    yield self._flush(batch)
                batch = []

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
                self._work_queue.task_done()
            elif event["type"] == ids.EVTYPE_EE_SNAPSHOT_UPDATE:
                iter_ = event.data["iter"]
                if iter_ not in self._iter_snapshot:
                    raise OutOfOrderSnapshotUpdateException(
                        f"got {ids.EVTYPE_EE_SNAPSHOT_UPDATE} without having stored snapshot for iter {iter_}"
                    )
                batch.append(event)
            else:
                raise ValueError("got unexpected event type", event["type"])
        if batch:
            yield self._flush(batch)

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
            wait_for_ws(self._monitor_url, self._token, self._cert, 2)
        except ConnectionRefusedError as e:
            logger.warning(f"{__name__} - exception {e}")
            return

        with create_ee_monitor(
            self._monitor_host,
            self._monitor_port,
            token=self._token,
            cert=self._cert,
            protocol=self._protocol,
        ) as monitor:
            for e in monitor.track():
                monitor.signal_cancel()
                break
        while self._drainer_thread.is_alive():
            self._clear_work_queue()
            time.sleep(1)
