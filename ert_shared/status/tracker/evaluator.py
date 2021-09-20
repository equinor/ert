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
import asyncio

from aiohttp import ClientError

from ert_shared.models.base_run_model import BaseRunModel
import ert_shared.ensemble_evaluator.entity.identifiers as ids
from ert_shared.ensemble_evaluator.entity.snapshot import PartialSnapshot, Snapshot
from ert_shared.ensemble_evaluator.monitor import create as create_ee_monitor
from ert_shared.ensemble_evaluator.utils import wait_for_evaluator
from ert_shared.status.entity.event import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)


class OutOfOrderSnapshotUpdateException(ValueError):
    pass


class EvaluatorTracker:
    DONE = "done"

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
        asyncio.set_event_loop(asyncio.new_event_loop())
        drainer_logger = logging.getLogger("ert_shared.ensemble_evaluator.drainer")
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

            except (ConnectionRefusedError, ClientError) as e:
                if not self._model.isFinished():
                    drainer_logger.debug(f"connection refused: {e}")

        drainer_logger.debug(
            "observed that model was finished, waiting tasks completion..."
        )
        # The model has finished, we indicate this by sending a DONE
        self._work_queue.put(EvaluatorTracker.DONE)
        self._work_queue.join()
        drainer_logger.debug("tasks complete")
        self._model.teardown_context()

    def track(self):
        while True:
            event = self._work_queue.get()
            if isinstance(event, str):
                try:
                    if event == EvaluatorTracker.DONE:
                        yield EndEvent(
                            failed=self._model.hasRunFailed(),
                            failed_msg=self._model.getFailMessage(),
                        )
                except GeneratorExit:
                    # consumers may exit at this point, make sure the last
                    # task is marked as done
                    pass
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
                self._iter_snapshot[iter_].merge_event(partial)
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
            asyncio.get_event_loop().run_until_complete(
                wait_for_evaluator(
                    base_url=self._monitor_url,
                    token=self._token,
                    cert=self._cert,
                    timeout=5,
                )
            )
        except ClientError as e:
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
