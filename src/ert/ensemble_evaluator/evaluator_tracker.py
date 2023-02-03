import asyncio
import logging
import queue
import threading
import time
from typing import TYPE_CHECKING, Dict, Iterator, Union

from aiohttp import ClientError
from websockets.exceptions import ConnectionClosedError

from ert.async_utils import get_event_loop
from ert.ensemble_evaluator.identifiers import (
    EVTYPE_EE_SNAPSHOT,
    EVTYPE_EE_SNAPSHOT_UPDATE,
    EVTYPE_EE_TERMINATED,
    STATUS,
)
from ert.ensemble_evaluator.state import (
    ENSEMBLE_STATE_CANCELLED,
    ENSEMBLE_STATE_FAILED,
    ENSEMBLE_STATE_STOPPED,
    REALIZATION_STATE_FAILED,
    REALIZATION_STATE_FINISHED,
)

from ._wait_for_evaluator import wait_for_evaluator
from .evaluator_connection_info import EvaluatorConnectionInfo
from .event import EndEvent, FullSnapshotEvent, SnapshotUpdateEvent
from .monitor import Monitor
from .snapshot import PartialSnapshot, Snapshot

if TYPE_CHECKING:
    from cloudevents.http.event import CloudEvent

    from ert.shared.models.base_run_model import BaseRunModel


class OutOfOrderSnapshotUpdateException(ValueError):
    pass


class EvaluatorTracker:
    DONE = "done"

    def __init__(
        self,
        model: "BaseRunModel",
        ee_con_info: EvaluatorConnectionInfo,
        next_ensemble_evaluator_wait_time: int = 5,
    ):
        self._model = model
        self._ee_con_info = ee_con_info
        self._next_ensemble_evaluator_wait_time = next_ensemble_evaluator_wait_time
        self._work_queue: "queue.Queue[CloudEvent]" = queue.Queue()
        self._drainer_thread = threading.Thread(
            target=self._drain_monitor,
            name="DrainerThread",
        )
        self._drainer_thread.start()
        self._iter_snapshot: Dict[int, Snapshot] = {}

    def _drain_monitor(self) -> None:
        asyncio.set_event_loop(asyncio.new_event_loop())
        drainer_logger = logging.getLogger("ert.ensemble_evaluator.drainer")
        while not self._model.isFinished():
            try:
                drainer_logger.debug("connecting to new monitor...")
                with Monitor(self._ee_con_info) as monitor:
                    drainer_logger.debug("connected")
                    for event in monitor.track():
                        if event["type"] in (
                            EVTYPE_EE_SNAPSHOT,
                            EVTYPE_EE_SNAPSHOT_UPDATE,
                        ):
                            self._work_queue.put(event)
                            if event.data.get(STATUS) in [
                                ENSEMBLE_STATE_STOPPED,
                                ENSEMBLE_STATE_FAILED,
                            ]:
                                drainer_logger.debug(
                                    "observed evaluation stopped event, signal done"
                                )
                                monitor.signal_done()
                            if event.data.get(STATUS) == ENSEMBLE_STATE_CANCELLED:
                                drainer_logger.debug(
                                    "observed evaluation cancelled event, exit drainer"
                                )
                                # Allow track() to emit an EndEvent.
                                self._work_queue.put(EvaluatorTracker.DONE)
                                return
                        elif event["type"] == EVTYPE_EE_TERMINATED:
                            drainer_logger.debug("got terminator event")
                # This sleep needs to be there. Refer to issue #1250: `Authority
                # on information about evaluations/experiments`
                time.sleep(self._next_ensemble_evaluator_wait_time)
            except (ConnectionRefusedError, ClientError) as e:
                if not self._model.isFinished():
                    drainer_logger.debug(f"connection refused: {e}")
            except ConnectionClosedError as e:
                # The monitor connection closed unexpectedly
                drainer_logger.debug(f"connection closed error: {e}")
            except BaseException:  # pylint: disable=broad-except
                drainer_logger.exception("unexpected error: ")
                # We really don't know what happened...  shut down
                # the thread and get out of here. The monitor has
                # been stopped by the ctx-mgr
                self._work_queue.put(EvaluatorTracker.DONE)
                self._work_queue.join()
                return
        drainer_logger.debug(
            "observed that model was finished, waiting tasks completion..."
        )
        # The model has finished, we indicate this by sending a DONE
        self._work_queue.put(EvaluatorTracker.DONE)
        self._work_queue.join()
        drainer_logger.debug("tasks complete")

    def track(
        self,
    ) -> Iterator[Union[FullSnapshotEvent, SnapshotUpdateEvent, EndEvent]]:
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
            if event["type"] == EVTYPE_EE_SNAPSHOT:
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
            elif event["type"] == EVTYPE_EE_SNAPSHOT_UPDATE:
                iter_ = event.data["iter"]
                if iter_ not in self._iter_snapshot:
                    raise OutOfOrderSnapshotUpdateException(
                        f"got {EVTYPE_EE_SNAPSHOT_UPDATE} without having stored "
                        f"snapshot for iter {iter_}"
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

    def is_finished(self) -> bool:
        return not self._drainer_thread.is_alive()

    def _progress(self) -> float:
        """Fraction of completed iterations over total iterations"""

        if self.is_finished():
            return 1.0
        elif not self._iter_snapshot:
            return 0.0
        else:
            # Calculate completed realizations
            current_iter = max(list(self._iter_snapshot.keys()))
            done_reals = 0
            all_reals = self._iter_snapshot[current_iter].reals
            if not all_reals:
                # Empty ensemble or all realizations deactivated
                return 1.0
            for real in all_reals.values():
                if real.status in [
                    REALIZATION_STATE_FINISHED,
                    REALIZATION_STATE_FAILED,
                ]:
                    done_reals += 1
            real_progress = float(done_reals) / len(all_reals)

            return (
                (current_iter + real_progress) / self._model.phaseCount()
                if self._model.phaseCount() != 1
                else real_progress
            )

    def _clear_work_queue(self) -> None:
        try:
            while True:
                self._work_queue.get_nowait()
                self._work_queue.task_done()
        except queue.Empty:
            pass

    def request_termination(self) -> None:
        logger = logging.getLogger("ert.ensemble_evaluator.tracker")
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
            logger.debug("requesting termination...")
            get_event_loop().run_until_complete(
                wait_for_evaluator(
                    base_url=self._ee_con_info.url,
                    token=self._ee_con_info.token,
                    cert=self._ee_con_info.cert,
                    timeout=5,
                )
            )
            logger.debug("requested termination")
        except ClientError as e:
            logger.warning(f"{__name__} - exception {e}")
            return

        with Monitor(self._ee_con_info) as monitor:
            monitor.signal_cancel()
        while self._drainer_thread.is_alive():
            self._clear_work_queue()
            time.sleep(1)
