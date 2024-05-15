from __future__ import annotations

import copy
import logging
import os
import shutil
import time
import uuid
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
from ert.analysis.event import AnalysisErrorEvent
from ert.cli import MODULE_MODE
from ert.config import ErtConfig, HookRuntime, QueueSystem
from ert.enkf_main import EnKFMain, _seed_sequence, create_run_path
from ert.ensemble_evaluator import (
    Ensemble,
    EnsembleBuilder,
    EnsembleEvaluator,
    EvaluatorServerConfig,
    Monitor,
    RealizationBuilder,
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
from ert.run_context import RunContext
from ert.runpaths import Runpaths
from ert.storage import Storage

from .event import (
    RunModelErrorEvent,
    RunModelStatusEvent,
    RunModelTimeEvent,
    RunModelUpdateBeginEvent,
    RunModelUpdateEndEvent,
)

event_logger = logging.getLogger("ert.event_log")

if TYPE_CHECKING:
    from ert.config import QueueConfig
    from ert.run_models.run_arguments import RunArgumentsType

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
        simulation_arguments: RunArgumentsType,
        config: ErtConfig,
        storage: Storage,
        queue_config: QueueConfig,
        status_queue: SimpleQueue[StatusEvents],
        phase_count: int = 1,
    ):
        """

        Parameters
        ----------
        simulation_arguments : Parameters for running the experiment,
            eg. activate realizations, analysis module
        queue_config : QueueConfig
        phase_count : Optional[int], optional
            Number of data assimilation cycles / iterations an experiment will have,
            by default 1
        """
        self._phase: int = 0
        self._phase_count = phase_count
        self._phase_name: str = "Starting..."

        self.start_time: Optional[int] = None
        self.stop_time: Optional[int] = None
        self._indeterminate: bool = False
        self._failed: bool = False
        self._exception: Optional[Exception] = None
        self._error_messages: MutableSequence[str] = []
        self._queue_config: QueueConfig = queue_config
        self._initial_realizations_mask: List[bool] = []
        self._completed_realizations_mask: List[bool] = []
        self.support_restart: bool = True
        self.ert_config = config
        self.ert = EnKFMain(config)
        self.facade = LibresFacade(self.ert)
        self._storage = storage
        self._simulation_arguments = simulation_arguments
        self.reset()
        # mapping from iteration number to ensemble id
        self._iter_map: Dict[int, str] = {}
        self.validate()
        self._context_env_keys: List[str] = []
        self.rng = np.random.default_rng(
            _seed_sequence(simulation_arguments.random_seed)
        )
        self.substitution_list = config.substitution_list
        self.run_paths = Runpaths(
            jobname_format=config.model_config.jobname_format_string,
            runpath_format=config.model_config.runpath_format_string,
            filename=str(config.runpath_file),
            substitution_list=self.substitution_list,
        )
        if hasattr(self.simulation_arguments, "current_ensemble"):
            current_ensemble = self.simulation_arguments.current_ensemble
            if current_ensemble is not None:
                self.run_paths.set_ert_ensemble(current_ensemble)

        self._iter_snapshot: Dict[int, Snapshot] = {}
        self._status_queue = status_queue
        self._end_queue: SimpleQueue[str] = SimpleQueue()

    def send_event(self, event: StatusEvents) -> None:
        self._status_queue.put(event)

    def send_smoother_event(self, iteration: int, event: AnalysisEvent) -> None:
        if isinstance(event, AnalysisStatusEvent):
            self.send_event(RunModelStatusEvent(iteration=iteration, msg=event.msg))
        elif isinstance(event, AnalysisTimeEvent):
            self.send_event(
                RunModelTimeEvent(
                    iteration=iteration,
                    elapsed_time=event.elapsed_time,
                    remaining_time=event.remaining_time,
                )
            )
        elif isinstance(event, AnalysisErrorEvent):
            self.send_event(
                RunModelErrorEvent(
                    iteration=iteration,
                    error_msg=event.error_msg,
                    smoother_snapshot=event.smoother_snapshot,
                )
            )

    @property
    def queue_system(self) -> QueueSystem:
        return self._queue_config.queue_system

    @property
    def simulation_arguments(self) -> RunArgumentsType:
        return self._simulation_arguments

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

    def restart(self) -> None:
        active_realizations = self._create_mask_from_failed_realizations()
        self._simulation_arguments.active_realizations = active_realizations
        self._simulation_arguments.prev_successful_realizations += (
            self._count_successful_realizations()
        )

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

    def _count_successful_realizations(self) -> int:
        """
        Counts the realizations completed in the previous ensemble run
        :return:
        """
        completed = self._completed_realizations_mask
        return completed.count(True)

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
        self, evaluator_server_config: EvaluatorServerConfig
    ) -> None:
        try:
            self.start_time = int(time.time())
            with captured_logs(self._error_messages):
                self._set_default_env_context()
                self._initial_realizations_mask = (
                    self._simulation_arguments.active_realizations
                )
                run_context = self.run_experiment(
                    evaluator_server_config=evaluator_server_config,
                )
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
        self, evaluator_server_config: EvaluatorServerConfig
    ) -> RunContext:
        raise NotImplementedError("Method must be implemented by inheritors!")

    def phaseCount(self) -> int:
        return self._phase_count

    def setPhaseCount(self, phase_count: int) -> None:
        self._phase_count = phase_count
        self.setPhase(0, "")

    def currentPhase(self) -> int:
        return self._phase

    def setPhaseName(
        self, phase_name: str, indeterminate: Optional[bool] = None
    ) -> None:
        self._phase_name = phase_name
        self.setIndeterminate(indeterminate)

    def getPhaseName(self) -> str:
        return self._phase_name

    def setIndeterminate(self, indeterminate: Optional[bool]) -> None:
        if indeterminate is not None:
            self._indeterminate = indeterminate

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

    def setPhase(
        self, phase: int, phase_name: str, indeterminate: Optional[bool] = None
    ) -> None:
        self.setPhaseName(phase_name)
        if not 0 <= phase <= self._phase_count:
            raise ValueError(
                f"Phase must be integer between (inclusive) 0 and {self._phase_count}"
            )

        self.setIndeterminate(indeterminate)

        if phase == self._phase_count:
            self._simulationEnded()

        self._phase = phase

    def get_runtime(self) -> Union[int, float]:
        if self.start_time is None:
            return 0
        elif self.stop_time is None:
            return time.time() - self.start_time
        return self.stop_time - self.start_time

    def isIndeterminate(self) -> bool:
        return not self.isFinished() and self._indeterminate

    @staticmethod
    def checkHaveSufficientRealizations(
        num_successful_realizations: int, minimum_realizations: int
    ) -> None:
        if num_successful_realizations == 0:
            raise ErtRunError("Experiment failed! All realizations failed!")
        if num_successful_realizations < minimum_realizations:
            raise ErtRunError(
                f"Too many realizations have failed! Number of successful realizations:"
                f" {num_successful_realizations}, expected minimal number of successful"
                f" realizations: {minimum_realizations}\n You can add/adjust "
                f"MIN_REALIZATIONS to allow (more) failures in your experiments."
            )

    def _progress(self) -> float:
        """Fraction of completed iterations over total iterations"""

        if self.isFinished():
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
                (current_iter + real_progress) / self.phaseCount()
                if self.phaseCount() != 1
                else real_progress
            )

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
            self.send_event(
                FullSnapshotEvent(
                    phase_name=self.getPhaseName(),
                    current_phase=self.currentPhase(),
                    total_phases=self.phaseCount(),
                    indeterminate=self.isIndeterminate(),
                    progress=self._progress(),
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
            self.send_event(
                SnapshotUpdateEvent(
                    phase_name=self.getPhaseName(),
                    current_phase=self.currentPhase(),
                    total_phases=self.phaseCount(),
                    indeterminate=self.isIndeterminate(),
                    progress=self._progress(),
                    iteration=iter_,
                    partial_snapshot=partial,
                )
            )

    async def run_monitor(self, ee_config: EvaluatorServerConfig) -> bool:
        try:
            event_logger.debug("connecting to new monitor...")
            async with Monitor(ee_config.get_connection_info()) as monitor:
                event_logger.debug("connected")
                async for event in monitor.track():
                    if event["type"] in (
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
                    elif event["type"] == EVTYPE_EE_TERMINATED:
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
    ) -> "Ensemble":
        builder = EnsembleBuilder().set_legacy_dependencies(
            self._queue_config,
            self._simulation_arguments.stop_long_running,
            self._simulation_arguments.minimum_required_realizations,
        )

        for iens, run_arg in enumerate(run_context):
            active = run_context.is_active(iens)
            real = RealizationBuilder().set_iens(iens).active(active)
            if active:
                real.set_forward_models(self.ert_config.forward_model_steps)
                real.set_max_runtime(self.ert_config.analysis_config.max_runtime)
                real.set_run_arg(run_arg)
                real.set_num_cpu(self.ert_config.preferred_num_cpu)
                real.set_job_script(self.ert_config.queue_config.job_script)
            builder.add_realization(real)
        return builder.set_id(str(uuid.uuid1()).split("-", maxsplit=1)[0]).build()

    @property
    def paths(self) -> List[str]:
        run_paths = []
        start_iteration = self._simulation_arguments.start_iteration
        number_of_iterations = self.simulation_arguments.num_iterations
        active_mask = self._simulation_arguments.active_realizations
        active_realizations = [i for i in range(len(active_mask)) if active_mask[i]]
        for iteration in range(start_iteration, number_of_iterations):
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

    def rm_run_path(self) -> None:
        for run_path in self.paths:
            if Path(run_path).exists():
                shutil.rmtree(run_path)

    def validate(self) -> None:
        errors = []

        active_mask = self._simulation_arguments.active_realizations
        active_realizations_count = len(
            [i for i in range(len(active_mask)) if active_mask[i]]
        )

        min_realization_count = self._simulation_arguments.minimum_required_realizations

        if active_realizations_count < min_realization_count:
            raise ValueError(
                f"Number of active realizations ({active_realizations_count}) is less "
                f"than the specified MIN_REALIZATIONS in the config file "
                f"({min_realization_count})"
            )

        current_ensemble = self._simulation_arguments.current_ensemble
        target_ensemble = self._simulation_arguments.target_ensemble

        if current_ensemble is not None:
            try:
                ensemble = self._storage.get_ensemble_by_name(current_ensemble)
                if ensemble.ensemble_size != self._simulation_arguments.ensemble_size:
                    errors.append(
                        f"- Existing ensemble: '{current_ensemble}' was created with a "
                        f"different ensemble size than the one specified in the ert "
                        f"configuration file \n ({ensemble.ensemble_size} "
                        f" != {self._simulation_arguments.ensemble_size})"
                    )
            except KeyError:
                pass
        if target_ensemble is not None:
            if target_ensemble == current_ensemble:
                errors.append(
                    f"- Target ensemble and current ensemble can not have the same name. "
                    f"They were both: {current_ensemble}"
                )

            if "%d" in target_ensemble:
                num_iterations = self._simulation_arguments.num_iterations
                for i in range(num_iterations):
                    try:
                        self._storage.get_ensemble_by_name(
                            target_ensemble % i  # noqa: S001
                        )
                        errors.append(
                            f"- Target ensemble: {target_ensemble % i} exists"  # noqa: S001
                        )
                    except KeyError:
                        pass
            else:
                try:
                    self._storage.get_ensemble_by_name(target_ensemble)
                    errors.append(f"- Target ensemble: {target_ensemble} exists")
                except KeyError:
                    pass
        if errors:
            raise ValueError("\n".join(errors))

    def _evaluate_and_postprocess(
        self,
        run_context: RunContext,
        evaluator_server_config: EvaluatorServerConfig,
    ) -> int:
        iteration = run_context.iteration

        phase_string = f"Running simulation for iteration: {iteration}"
        self.setPhase(iteration, phase_string, indeterminate=False)
        create_run_path(run_context, self.ert_config)

        phase_string = f"Pre processing for iteration: {iteration}"
        self.setPhaseName(phase_string, indeterminate=True)
        self.ert.runWorkflows(
            HookRuntime.PRE_SIMULATION, self._storage, run_context.ensemble
        )

        phase_string = f"Running forecast for iteration: {iteration}"
        self.setPhaseName(phase_string, indeterminate=False)

        successful_realizations = self.run_ensemble_evaluator(
            run_context, evaluator_server_config
        )
        starting_realizations = run_context.active_realizations
        failed_realizations = list(
            set(starting_realizations) - set(successful_realizations)
        )
        for iens in failed_realizations:
            run_context.deactivate_realization(iens)

        num_successful_realizations = len(successful_realizations)
        num_successful_realizations += (
            self._simulation_arguments.prev_successful_realizations
        )
        self.checkHaveSufficientRealizations(
            num_successful_realizations,
            self._simulation_arguments.minimum_required_realizations,
        )

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
        self.setPhaseName(phase_string, indeterminate=True)
        self.ert.runWorkflows(
            HookRuntime.POST_SIMULATION, self._storage, run_context.ensemble
        )

        return num_successful_realizations
