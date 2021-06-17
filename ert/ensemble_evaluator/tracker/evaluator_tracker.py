import asyncio
from typing import Dict
from ert_shared.async_utils import get_event_loop
from ert_shared.status.entity.state import (
    ENSEMBLE_STATE_CANCELLED,
    ENSEMBLE_STATE_STOPPED,
    ENSEMBLE_STATE_FAILED,
)
import logging
import queue
import threading
import time
from typing import Generator, Union, Dict, TYPE_CHECKING

from aiohttp import ClientError
from websockets.exceptions import ConnectionClosedError

from ert.ensemble_evaluator.evaluator_connection_info import EvaluatorConnectionInfo

import ert_shared.ensemble_evaluator.entity.identifiers as ids
from ert_shared.async_utils import get_event_loop
from ert_shared.ensemble_evaluator.entity.snapshot import PartialSnapshot, Snapshot
from ert_shared.ensemble_evaluator.monitor import create as create_ee_monitor
from ert_shared.ensemble_evaluator.monitor import (
    create_experiment as create_experiment_monitor,
)
from ert_shared.ensemble_evaluator.utils import (
    wait_for_evaluator,
    get_current_evaluations,
)
from ert_shared.status.entity.event import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)
from ert_shared.status.entity.state import (
    ENSEMBLE_STATE_CANCELLED,
    ENSEMBLE_STATE_STOPPED,
    ENSEMBLE_STATE_FAILED,
)
from ert_shared.status.utils import state

if TYPE_CHECKING:
    from ert3.evaluator._evaluator import ERT3RunModel
    from ert_shared.models.base_run_model import BaseRunModel
    from cloudevents.http.event import CloudEvent  # type: ignore


class OutOfOrderSnapshotUpdateException(ValueError):
    pass


class EvaluatorTracker:
    DONE = "done"

    def __init__(
        self,
        model: Union["BaseRunModel", "ERT3RunModel"],
        ee_con_info: EvaluatorConnectionInfo,
    ):
        self._model = model
        self._ee_con_info = ee_con_info
        self._work_queue: "queue.Queue[CloudEvent]" = queue.Queue()
        self._last_eval_id = None

        self._drainer_thread = threading.Thread(
            target=self._drain_monitor,
            name="DrainerThread",
        )
        self._drainer_thread.start()
        self._iter_snapshot: Dict[int, Snapshot] = {}

    def _track_evaluation(self, evaluation_id, logger):
        try:
            logger.debug("connecting to new monitor...")
            with create_ee_monitor(
                evaluation_id,
                self._ee_con_info,
            ) as monitor:
                logger.debug("connected")
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
                            logger.debug(
                                "observed evaluation stopped event, signal done"
                            )
                            monitor.signal_done()
                        if event.data.get(ids.STATUS) == ENSEMBLE_STATE_CANCELLED:
                            logger.debug(
                                "observed evaluation cancelled event, exit drainer"
                            )
                            # Allow track() to emit an EndEvent.
                            self._work_queue.put(EvaluatorTracker.DONE)
                            return
                    elif event["type"] == ids.EVTYPE_EE_TERMINATED:
                        logger.debug("got terminator event")

        except (ConnectionRefusedError, ClientError) as e:
            logger.debug(f"connection error: {e}")
        except (ConnectionClosedError) as e:
            # The monitor connection closed unexpectedly
            logger.debug(f"connection closed error: {e}")

    def _drain_monitor(self):
        asyncio.set_event_loop(asyncio.new_event_loop())
        drainer_logger = logging.getLogger("ert_shared.ensemble_evaluator.drainer")
        tracked_evaluations: Dict[str, threading.Thread] = dict()
        while self._model.experiment_id is None:
            time.sleep(0.1)
        with create_experiment_monitor(
            self._model.experiment_id, self._ee_con_info
        ) as monitor:
            for event in monitor.track():
                if not event.data:
                    continue
                not_tracked_yet = set(event.data["evaluations"]) - set(
                    tracked_evaluations.keys()
                )
                for evaluation_id in not_tracked_yet:
                    tracked_evaluations[evaluation_id] = threading.Thread(
                        target=self._track_evaluation,
                        args=(evaluation_id, drainer_logger),
                        name=f"TrackEvaluationThread-{evaluation_id}",
                    )
                    tracked_evaluations[evaluation_id].start()
                    self._last_eval_id = evaluation_id
            for t in tracked_evaluations.values():
                t.join()

        drainer_logger.debug(
            "observed that model was finished, waiting tasks completion..."
        )
        # The model has finished, we indicate this by sending a DONE
        self._work_queue.put(EvaluatorTracker.DONE)
        self._work_queue.join()
        drainer_logger.debug("tasks complete")
        self._model.teardown_context()

    def track(
        self,
    ) -> Generator[Union[FullSnapshotEvent, SnapshotUpdateEvent, EndEvent], None, None]:
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
            if event["type"] == ids.EVTYPE_EE_SNAPSHOT:
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
                        f"got {ids.EVTYPE_EE_SNAPSHOT_UPDATE} without having stored snapshot"
                        f"for iter {iter_}"
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
            for real in self._iter_snapshot[current_iter].get_reals().values():
                if real.status in [
                    state.REALIZATION_STATE_FINISHED,
                    state.REALIZATION_STATE_FAILED,
                ]:
                    done_reals += 1
            total_reals = len(self._iter_snapshot[current_iter].get_reals())
            real_progress = float(done_reals) / total_reals
            return (current_iter + real_progress) / self._model.phaseCount()

    def _clear_work_queue(self) -> None:
        try:
            while True:
                self._work_queue.get_nowait()
                self._work_queue.task_done()
        except queue.Empty:
            pass

    def request_termination(self) -> None:
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
            get_event_loop().run_until_complete(
                wait_for_evaluator(
                    base_url=self._ee_con_info.url,
                    token=self._ee_con_info.token,
                    cert=self._ee_con_info.cert,
                    timeout=5,
                )
            )
        except ClientError as e:
            logger.warning(f"{__name__} - exception {e}")
            return

        # TODO: Fix last_eval_id somehow
        with create_ee_monitor(self._last_eval_id, self._ee_con_info) as monitor:
            monitor.signal_cancel()
        while self._drainer_thread.is_alive():
            self._clear_work_queue()
            time.sleep(1)
