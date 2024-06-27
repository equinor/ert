from __future__ import annotations

import copy
import logging
import os
import shutil
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from queue import SimpleQueue
from typing import (
    TYPE_CHECKING,
    Dict,
    Generator,
    List,
    MutableSequence,
    Optional,
    Type,
    Union,
)

import numpy as np
from cloudevents.http import CloudEvent

from _ert.async_utils import get_running_loop
from ert.analysis import AnalysisEvent, AnalysisStatusEvent, AnalysisTimeEvent
from ert.analysis.event import (
    AnalysisCompleteEvent,
    AnalysisDataEvent,
    AnalysisErrorEvent,
)
from ert.config import ErtConfig, HookRuntime, QueueSystem
from ert.enkf_main import _seed_sequence, create_run_path
from ert.ensemble_evaluator import Ensemble as EEEnsemble
from ert.ensemble_evaluator import (
    EnsembleEvaluator,
    EvaluatorServerConfig,
    Monitor,
    Realization,
)
from ert.ensemble_evaluator.event import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)
from ert.ensemble_evaluator.identifiers import (
    EVTYPE_EE_SNAPSHOT,
    EVTYPE_EE_SNAPSHOT_UPDATE,
    EVTYPE_EE_TERMINATED,
    STATUS,
)
from ert.ensemble_evaluator.snapshot import PartialSnapshot, Snapshot
from ert.ensemble_evaluator.state import (
    ENSEMBLE_STATE_CANCELLED,
    ENSEMBLE_STATE_FAILED,
    ENSEMBLE_STATE_STOPPED,
    REALIZATION_STATE_FAILED,
    REALIZATION_STATE_FINISHED,
)
from ert.libres_facade import LibresFacade
from ert.mode_definitions import MODULE_MODE
from ert.run_context import RunContext
from ert.runpaths import Runpaths
from ert.storage import Ensemble, Storage
from ert.workflow_runner import WorkflowRunner

from .event import (
    RunModelDataEvent,
    RunModelErrorEvent,
    RunModelStatusEvent,
    RunModelTimeEvent,
    RunModelUpdateBeginEvent,
    RunModelUpdateEndEvent,
)

event_logger = logging.getLogger("ert.event_log")

if TYPE_CHECKING:
    from ert.config import QueueConfig

StatusEvents = Union[
    FullSnapshotEvent,
    SnapshotUpdateEvent,
    EndEvent,
    AnalysisEvent,
    AnalysisStatusEvent,
    AnalysisTimeEvent,
    RunModelErrorEvent,
    RunModelStatusEvent,
    RunModelTimeEvent,
    RunModelUpdateBeginEvent,
    RunModelDataEvent,
    RunModelUpdateEndEvent,
]


class OutOfOrderSnapshotUpdateException(ValueError):
    pass


class ErtRunError(Exception):
    pass


class _LogAggregration(logging.Handler):
    def __init__(self, messages: MutableSequence[str]) -> None:
        self.messages = messages

        # Contains list of record names that should be exlucded from aggregated logs
        self.exclude_logs: List[str] = []
        super().__init__()

    def emit(self, record: logging.LogRecord) -> None:
        if record.name not in self.exclude_logs:
            self.messages.append(record.getMessage())


@contextmanager
def captured_logs(
    messages: MutableSequence[str], level: int = logging.ERROR
) -> Generator[None, None, None]:
    handler = _LogAggregration(messages)
    root_logger = logging.getLogger()
    handler.setLevel(level)
    root_logger.addHandler(handler)
    try:
        yield
    finally:
        root_logger.removeHandler(handler)


class BaseRunModel:
    def __init__(
        self,
        config: ErtConfig,
        storage: Storage,
        queue_config: QueueConfig,
        status_queue: SimpleQueue[StatusEvents],
        active_realizations: List[bool],
        phase_count: int = 1,
        number_of_iterations: int = 1,
        start_iteration: int = 0,
        random_seed: Optional[int] = None,
        minimum_required_realizations: int = 0,
    ):
        """
        BaseRunModel serves as the base class for the various experiment modes,
        and contains logic for interacting with the Ensemble Evaluator by running
        the forward model and passing events back through the supplied queue.
        """
        self._phase: int = 0
        self._phase_count = phase_count
        self._phase_name: str = "Starting..."

        self.start_time: Optional[int] = None
        self.stop_time: Optional[int] = None
        self._failed: bool = False
        self._exception: Optional[Exception] = None
        self._error_messages: MutableSequence[str] = []
        self._queue_config: QueueConfig = queue_config
        self._initial_realizations_mask: List[bool] = copy.copy(active_realizations)
        self._completed_realizations_mask: List[bool] = []
        self.support_restart: bool = True
        self.ert_config = config
        self.facade = LibresFacade(self.ert_config)
        self._storage = storage
        self._context_env_keys: List[str] = []
        self.random_seed: int = _seed_sequence(random_seed)
        self.rng = np.random.default_rng(self.random_seed)
        self.substitution_list = config.substitution_list

        self.run_paths = Runpaths(
            jobname_format=config.model_config.jobname_format_string,
            runpath_format=config.model_config.runpath_format_string,
            filename=str(config.runpath_file),
            substitution_list=self.substitution_list,
        )
        self._iter_snapshot: Dict[int, Snapshot] = {}
        self._status_queue = status_queue
        self._end_queue: SimpleQueue[str] = SimpleQueue()
        # This holds state about the run model
        self.minimum_required_realizations = minimum_required_realizations
        self.active_realizations = copy.copy(active_realizations)
        self.number_of_iterations = number_of_iterations
        self.start_iteration = start_iteration
        self.validate()

    @classmethod
    def name(cls) -> str:
        return "Base run model"

    def send_event(self, event: StatusEvents) -> None:
        self._status_queue.put(event)

    def send_smoother_event(
        self, iteration: int, run_id: uuid.UUID, event: AnalysisEvent
    ) -> None:
        if isinstance(event, AnalysisStatusEvent):
            self.send_event(
                RunModelStatusEvent(iteration=iteration, run_id=run_id, msg=event.msg)
            )
        elif isinstance(event, AnalysisTimeEvent):
            self.send_event(
                RunModelTimeEvent(
                    iteration=iteration,
                    run_id=run_id,
                    elapsed_time=event.elapsed_time,
                    remaining_time=event.remaining_time,
                )
            )
        elif isinstance(event, AnalysisErrorEvent):
            self.send_event(
                RunModelErrorEvent(
                    iteration=iteration,
                    run_id=run_id,
                    error_msg=event.error_msg,
                    data=event.data,
                )
            )
        elif isinstance(event, AnalysisDataEvent):
            self.send_event(
                RunModelDataEvent(
                    iteration=iteration, run_id=run_id, name=event.name, data=event.data
                )
            )
        elif isinstance(event, AnalysisCompleteEvent):
            self.send_event(
                RunModelUpdateEndEvent(
                    iteration=iteration, run_id=run_id, data=event.data
                )
            )

    @property
    def queue_system(self) -> QueueSystem:
        return self._queue_config.queue_system

    @property
    def _ensemble_size(self) -> int:
        return len(self._initial_realizations_mask)

    def cancel(self) -> None:
        self._end_queue.put("END")

    def reset(self) -> None:
        self._failed = False
        self._error_messages = []
        self._exception = None
        self._phase = 0

    def has_failed_realizations(self) -> bool:
        return any(self._create_mask_from_failed_realizations())

    def _create_mask_from_failed_realizations(self) -> List[bool]:
        """
        Creates a list of bools representing the failed realizations,
        i.e., a realization that has failed is assigned a True value.
        """
        if self._completed_realizations_mask:
            return [
                initial and not completed
                for initial, completed in zip(
                    self._initial_realizations_mask, self._completed_realizations_mask
                )
            ]
        else:
            # If all realisations fail
            return [True] * len(self._initial_realizations_mask)

    def set_env_key(self, key: str, value: str) -> None:
        """
        Will set an environment variable that will be available until the
        model run ends.
        """
        self._context_env_keys.append(key)
        os.environ[key] = value

    def _set_default_env_context(self) -> None:
        """
        Set some default environment variables that need to be
        available while the model is running
        """
        simulation_mode = MODULE_MODE.get(type(self).__name__, "")
        self.set_env_key("_ERT_SIMULATION_MODE", simulation_mode)

    def _clean_env_context(self) -> None:
        """
        Clean all previously environment variables set using set_env_key
        """
        for key in self._context_env_keys:
            os.environ.pop(key, None)
        self._context_env_keys = []

    def start_simulations_thread(
        self,
        evaluator_server_config: EvaluatorServerConfig,
        restart: bool = False,
    ) -> None:
        try:
            self.start_time = int(time.time())
            self.stop_time = None
            with captured_logs(self._error_messages):
                self._set_default_env_context()
                run_context = self.run_experiment(
                    evaluator_server_config=evaluator_server_config,
                    restart=restart,
                )
                if self._completed_realizations_mask:
                    combined = np.logical_or(
                        np.array(self._completed_realizations_mask),
                        np.array(run_context.mask),
                    )
                    self._completed_realizations_mask = list(combined)
                else:
                    self._completed_realizations_mask = run_context.mask
        except ErtRunError as e:
            self._completed_realizations_mask = []
            self._failed = True
            self._exception = e
            self._simulationEnded()
        except UserWarning as e:
            self._exception = e
            self._simulationEnded()
        except Exception as e:
            self._failed = True
            self._exception = e
            self._simulationEnded()

    def run_experiment(
        self,
        evaluator_server_config: EvaluatorServerConfig,
        restart: bool = False,
    ) -> RunContext:
        raise NotImplementedError("Method must be implemented by inheritors!")

    def phaseCount(self) -> int:
        return self._phase_count

    def setPhaseCount(self, phase_count: int) -> None:
        self._phase_count = phase_count
        self.setPhase(0, "")

    def currentPhase(self) -> int:
        return self._phase

    def setPhaseName(self, phase_name: str) -> None:
        self._phase_name = phase_name

    def getPhaseName(self) -> str:
        return self._phase_name

    def isFinished(self) -> bool:
        return self._phase == self._phase_count or self.hasRunFailed()

    def hasRunFailed(self) -> bool:
        return self._failed

    def getFailMessage(self) -> str:
        msg = "\n".join(self._error_messages)
        if self._exception is None:
            return msg
        return f"{self._exception}\n{msg}"

    def reraise_exception(self, exctype: Type[Exception]) -> None:
        """
        Re-raise an exception if it was set, otherwise return
        """
        if self._exception is not None:
            raise exctype(self.getFailMessage()).with_traceback(
                self._exception.__traceback__
            )

    def _simulationEnded(self) -> None:
        self._clean_env_context()
        self.stop_time = int(time.time())
        self.send_end_event()

    def setPhase(self, phase: int, phase_name: str) -> None:
        if not 0 <= phase <= self._phase_count:
            raise ValueError(
                f"Phase must be integer between (inclusive) 0 and {self._phase_count}"
            )

        self.setPhaseName(phase_name)

        if phase == self._phase_count:
            self._simulationEnded()

        self._phase = phase

    def get_runtime(self) -> int:
        if self.start_time is None:
            return 0
        elif self.stop_time is None:
            return round(time.time() - self.start_time)
        return self.stop_time - self.start_time

    def _current_status(self) -> tuple[dict[str, int], float, int]:
        current_iter = max(list(self._iter_snapshot.keys()))
        done_realizations = 0
        all_realizations = self._iter_snapshot[current_iter].reals
        current_progress = 0.0
        status: dict[str, int] = defaultdict(int)
        realization_count = len(all_realizations)

        if all_realizations:
            for real in all_realizations.values():
                status[str(real.status)] += 1

                if real.status in [
                    REALIZATION_STATE_FINISHED,
                    REALIZATION_STATE_FAILED,
                ]:
                    done_realizations += 1

            realization_progress = float(done_realizations) / len(all_realizations)
            current_progress = (
                (current_iter + realization_progress) / self.phaseCount()
                if self.phaseCount() != 1
                else realization_progress
            )

        return status, current_progress, realization_count

    def send_end_event(self) -> None:
        self.send_event(
            EndEvent(
                failed=self.hasRunFailed(),
                failed_msg=self.getFailMessage(),
            )
        )

    def send_snapshot_event(self, event: CloudEvent) -> None:
        if event["type"] == EVTYPE_EE_SNAPSHOT:
            iter_ = event.data["iter"]
            snapshot = Snapshot(event.data)
            self._iter_snapshot[iter_] = snapshot
            status, current_progress, realization_count = self._current_status()
            self.send_event(
                FullSnapshotEvent(
                    phase_name=self.getPhaseName(),
                    current_phase=self.currentPhase(),
                    total_phases=self.phaseCount(),
                    progress=current_progress,
                    realization_count=realization_count,
                    status_count=status,
                    iteration=iter_,
                    snapshot=copy.deepcopy(snapshot),
                )
            )
        elif event["type"] == EVTYPE_EE_SNAPSHOT_UPDATE:
            iter_ = event.data["iter"]
            if iter_ not in self._iter_snapshot:
                raise OutOfOrderSnapshotUpdateException(
                    f"got {EVTYPE_EE_SNAPSHOT_UPDATE} without having stored "
                    f"snapshot for iter {iter_}"
                )
            partial = PartialSnapshot(self._iter_snapshot[iter_]).from_cloudevent(event)
            self._iter_snapshot[iter_].merge_event(partial)
            status, current_progress, realization_count = self._current_status()
            self.send_event(
                SnapshotUpdateEvent(
                    phase_name=self.getPhaseName(),
                    current_phase=self.currentPhase(),
                    total_phases=self.phaseCount(),
                    progress=current_progress,
                    realization_count=realization_count,
                    status_count=status,
                    iteration=iter_,
                    partial_snapshot=partial,
                )
            )

    async def run_monitor(self, ee_config: EvaluatorServerConfig) -> bool:
        try:
            event_logger.debug("connecting to new monitor...")
            async with Monitor(ee_config.get_connection_info()) as monitor:
                event_logger.debug("connected")
                async for event in monitor.track(heartbeat_interval=0.1):
                    if event is not None and event["type"] in (
                        EVTYPE_EE_SNAPSHOT,
                        EVTYPE_EE_SNAPSHOT_UPDATE,
                    ):
                        self.send_snapshot_event(event)
                        if event.data.get(STATUS) in [
                            ENSEMBLE_STATE_STOPPED,
                            ENSEMBLE_STATE_FAILED,
                        ]:
                            event_logger.debug(
                                "observed evaluation stopped event, signal done"
                            )
                            await monitor.signal_done()

                        if event.data.get(STATUS) == ENSEMBLE_STATE_CANCELLED:
                            event_logger.debug(
                                "observed evaluation cancelled event, exit drainer"
                            )
                            # Allow track() to emit an EndEvent.
                            return False
                    elif event is not None and event["type"] == EVTYPE_EE_TERMINATED:
                        event_logger.debug("got terminator event")

                    if not self._end_queue.empty():
                        event_logger.debug("Run model canceled - during evaluation")
                        self._end_queue.get()
                        await monitor.signal_cancel()
                        event_logger.debug(
                            "Run model canceled - during evaluation - cancel sent"
                        )
        except BaseException:
            event_logger.exception("unexpected error: ")
            # We really don't know what happened...  shut down
            # the thread and get out of here. The monitor has
            # been stopped by the ctx-mgr
            return False

        return True

    def run_ensemble_evaluator(
        self, run_context: RunContext, ee_config: EvaluatorServerConfig
    ) -> List[int]:
        if not self._end_queue.empty():
            event_logger.debug("Run model canceled - pre evaluation")
            self._end_queue.get()
            return []
        ensemble = self._build_ensemble(run_context)
        evaluator = EnsembleEvaluator(
            ensemble,
            ee_config,
            run_context.iteration,
        )
        evaluator.start_running()

        if not get_running_loop().run_until_complete(self.run_monitor(ee_config)):
            return []

        event_logger.debug(
            "observed that model was finished, waiting tasks completion..."
        )
        # The model has finished, we indicate this by sending a DONE
        event_logger.debug("tasks complete")

        evaluator.join()
        if not self._end_queue.empty():
            event_logger.debug("Run model canceled - post evaluation")
            self._end_queue.get()
            return []
        return evaluator.get_successful_realizations()

    def _build_ensemble(
        self,
        run_context: RunContext,
    ) -> EEEnsemble:
        realizations = []
        for iens, run_arg in enumerate(run_context):
            realizations.append(
                Realization(
                    active=run_context.is_active(iens),
                    iens=iens,
                    forward_models=self.ert_config.forward_model_steps,
                    max_runtime=self.ert_config.analysis_config.max_runtime,
                    run_arg=run_arg,
                    num_cpu=self.ert_config.preferred_num_cpu,
                    job_script=self.ert_config.queue_config.job_script,
                )
            )
        return EEEnsemble(
            realizations,
            {},
            self._queue_config,
            self.minimum_required_realizations,
            str(run_context.ensemble.experiment.id),
        )

    @property
    def paths(self) -> List[str]:
        run_paths = []
        number_of_iterations = self.number_of_iterations
        active_realizations = np.where(self.active_realizations)[0]
        for iteration in range(
            self.start_iteration, self.start_iteration + number_of_iterations
        ):
            run_paths.extend(self.run_paths.get_paths(active_realizations, iteration))
        return run_paths

    def check_if_runpath_exists(self) -> bool:
        """
        Determine if the run_path exists by checking if it contains
        at least one iteration directory for the realizations in the active mask.
        The run_path can contain one or two %d specifiers ie:
            "realization-%d/iter-%d/"
            "realization-%d/"
        """
        return any(Path(run_path).exists() for run_path in self.paths)

    def get_number_of_existing_runpaths(self) -> int:
        return [Path(run_path).exists() for run_path in self.paths].count(True)

    def get_number_of_active_realizations(self) -> int:
        return self.active_realizations.count(True)

    def rm_run_path(self) -> None:
        for run_path in self.paths:
            if Path(run_path).exists():
                shutil.rmtree(run_path)

    def validate(self) -> None:
        active_realizations_count = self.get_number_of_active_realizations()
        min_realization_count = self.minimum_required_realizations

        if active_realizations_count < min_realization_count:
            raise ValueError(
                f"Number of active realizations ({active_realizations_count}) is less "
                f"than the specified MIN_REALIZATIONS"
                f"({min_realization_count})"
            )

    def run_workflows(
        self, runtime: HookRuntime, storage: Storage, ensemble: Ensemble
    ) -> None:
        for workflow in self.ert_config.hooked_workflows[runtime]:
            WorkflowRunner(
                workflow, storage, ensemble, ert_config=self.ert_config
            ).run_blocking()

    def _evaluate_and_postprocess(
        self,
        run_context: RunContext,
        evaluator_server_config: EvaluatorServerConfig,
    ) -> int:
        iteration = run_context.iteration

        phase_string = f"Running simulation for iteration: {iteration}"
        self.setPhase(iteration, phase_string)
        create_run_path(run_context, self.ert_config)

        phase_string = f"Pre processing for iteration: {iteration}"
        self.setPhaseName(phase_string)
        self.run_workflows(
            HookRuntime.PRE_SIMULATION, self._storage, run_context.ensemble
        )

        phase_string = f"Running forecast for iteration: {iteration}"
        self.setPhaseName(phase_string)

        successful_realizations = self.run_ensemble_evaluator(
            run_context, evaluator_server_config
        )
        starting_realizations = run_context.active_realizations
        failed_realizations = list(
            set(starting_realizations) - set(successful_realizations)
        )
        for iens in failed_realizations:
            run_context.deactivate_realization(iens)
            self.active_realizations[iens] = False

        num_successful_realizations = len(successful_realizations)
        self.validate()
        event_logger.info(
            f"Experiment ran on QUEUESYSTEM: {self._queue_config.queue_system}"
        )
        event_logger.info(
            f"Experiment ran with number of realizations: {self._ensemble_size}"
        )
        event_logger.info(
            f"Experiment run ended with number of realizations succeeding: {num_successful_realizations}"
        )
        event_logger.info(
            f"Experiment run ended with number of realizations failing: {self._ensemble_size - num_successful_realizations}"
        )
        event_logger.info(f"Experiment run finished in: {self.get_runtime()}s")

        phase_string = f"Post processing for iteration: {iteration}"
        self.setPhaseName(phase_string)
        self.run_workflows(
            HookRuntime.POST_SIMULATION, self._storage, run_context.ensemble
        )

        return num_successful_realizations
