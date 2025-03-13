from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import dataclasses
import functools
import logging
import os
import shutil
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Generator, MutableSequence
from contextlib import contextmanager
from pathlib import Path
from queue import SimpleQueue
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from _ert.events import EESnapshot, EESnapshotUpdate, EETerminated, Event
from ert.analysis import ErtAnalysisError, smoother_update
from ert.analysis.event import (
    AnalysisCompleteEvent,
    AnalysisDataEvent,
    AnalysisErrorEvent,
    AnalysisEvent,
)
from ert.config import (
    ESSettings,
    ForwardModelStep,
    HookRuntime,
    ModelConfig,
    QueueSystem,
    UpdateSettings,
    Workflow,
)
from ert.enkf_main import _seed_sequence, create_run_path
from ert.ensemble_evaluator import Ensemble as EEEnsemble
from ert.ensemble_evaluator import (
    EnsembleEvaluator,
    EvaluatorServerConfig,
    Monitor,
    Realization,
)
from ert.ensemble_evaluator.identifiers import STATUS
from ert.ensemble_evaluator.snapshot import EnsembleSnapshot
from ert.ensemble_evaluator.state import (
    ENSEMBLE_STATE_CANCELLED,
    ENSEMBLE_STATE_FAILED,
    ENSEMBLE_STATE_STOPPED,
    REALIZATION_STATE_FAILED,
    REALIZATION_STATE_FINISHED,
)
from ert.mode_definitions import MODULE_MODE
from ert.plugins import WorkflowFixtures
from ert.runpaths import Runpaths
from ert.storage import Ensemble, Storage
from ert.substitutions import Substitutions
from ert.trace import tracer
from ert.utils import log_duration
from ert.workflow_runner import WorkflowRunner

from ..run_arg import RunArg
from .event import (
    AnalysisStatusEvent,
    AnalysisTimeEvent,
    EndEvent,
    FullSnapshotEvent,
    RunModelDataEvent,
    RunModelErrorEvent,
    RunModelStatusEvent,
    RunModelTimeEvent,
    RunModelUpdateBeginEvent,
    RunModelUpdateEndEvent,
    SnapshotUpdateEvent,
    StatusEvents,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ert.config import QueueConfig


class OutOfOrderSnapshotUpdateException(ValueError):
    pass


class ErtRunError(Exception):
    pass


def delete_runpath(run_path: str) -> None:
    if os.path.exists(run_path):
        shutil.rmtree(run_path)


class UserCancelled(Exception):
    pass


class _LogAggregration(logging.Handler):
    def __init__(self, messages: MutableSequence[str]) -> None:
        self.messages = messages

        # Contains list of record names that should be excluded from aggregated logs
        self.exclude_logs: list[str] = []
        super().__init__()

    def emit(self, record: logging.LogRecord) -> None:
        if record.name not in self.exclude_logs:
            self.messages.append(record.getMessage())


@contextmanager
def captured_logs(
    messages: MutableSequence[str], level: int = logging.ERROR
) -> Generator[None]:
    handler = _LogAggregration(messages)
    root_logger = logging.getLogger()
    handler.setLevel(level)
    root_logger.addHandler(handler)
    try:
        yield
    finally:
        root_logger.removeHandler(handler)


class StartSimulationsThreadFn(Protocol):
    def __call__(
        self, evaluator_server_config: EvaluatorServerConfig, restart: bool = False
    ) -> None: ...


@dataclasses.dataclass
class BaseRunModelAPI:
    experiment_name: str
    queue_system: str
    runpath_format_string: str
    support_restart: bool
    start_simulations_thread: StartSimulationsThreadFn
    cancel: Callable[[], None]
    get_runtime: Callable[[], int]
    has_failed_realizations: Callable[[], bool]


class BaseRunModel(ABC):
    def __init__(
        self,
        storage: Storage,
        runpath_file: Path,
        user_config_file: Path,
        env_vars: dict[str, str],
        env_pr_fm_step: dict[str, dict[str, Any]],
        model_config: ModelConfig,
        queue_config: QueueConfig,
        forward_model_steps: list[ForwardModelStep],
        status_queue: SimpleQueue[StatusEvents],
        substitutions: Substitutions,
        templates: list[tuple[str, str]],
        hooked_workflows: defaultdict[HookRuntime, list[Workflow]],
        active_realizations: list[bool],
        log_path: Path,
        total_iterations: int = 1,
        start_iteration: int = 0,
        random_seed: int | None = None,
        minimum_required_realizations: int = 0,
    ):
        """
        BaseRunModel serves as the base class for the various experiment modes,
        and contains logic for interacting with the Ensemble Evaluator by running
        the forward model and passing events back through the supplied queue.
        """
        self._total_iterations = total_iterations
        self.start_time: int | None = None
        self.stop_time: int | None = None
        self._queue_config: QueueConfig = queue_config
        self._initial_realizations_mask: list[bool] = copy.copy(active_realizations)
        self._completed_realizations_mask: list[bool] = []
        self.support_restart: bool = True
        self._storage = storage
        self._context_env: dict[str, str] = {}
        self.random_seed: int = _seed_sequence(random_seed)
        self.rng = np.random.default_rng(self.random_seed)
        self._substitutions: Substitutions = substitutions
        self._model_config: ModelConfig = model_config
        self._runpath_file: Path = runpath_file
        self._forward_model_steps: list[ForwardModelStep] = forward_model_steps
        self._user_config_file: Path = user_config_file
        self._templates: list[tuple[str, str]] = templates
        self._hooked_workflows: defaultdict[HookRuntime, list[Workflow]] = (
            hooked_workflows
        )
        self._log_path = log_path

        self._env_vars: dict[str, str] = env_vars
        self._env_pr_fm_step: dict[str, dict[str, Any]] = env_pr_fm_step

        self.run_paths = Runpaths(
            jobname_format=self._model_config.jobname_format_string,
            runpath_format=self._model_config.runpath_format_string,
            filename=str(self._runpath_file),
            substitutions=self._substitutions,
            eclbase=self._model_config.eclbase_format_string,
        )
        self._iter_snapshot: dict[int, EnsembleSnapshot] = {}
        self._status_queue = status_queue
        self._end_queue: SimpleQueue[str] = SimpleQueue()
        # This holds state about the run model
        self.minimum_required_realizations = minimum_required_realizations
        self.active_realizations = copy.copy(active_realizations)
        self.start_iteration = start_iteration
        self.restart = False

    @property
    def api(self) -> BaseRunModelAPI:
        return BaseRunModelAPI(
            experiment_name=self.name(),
            queue_system=self._queue_config.queue_system,
            runpath_format_string=str(self._runpath_file),
            get_runtime=self.get_runtime,
            start_simulations_thread=self.start_simulations_thread,
            has_failed_realizations=self.has_failed_realizations,
            support_restart=self.support_restart,
            cancel=self.cancel,
        )

    def reports_dir(self, experiment_name: str) -> str:
        return str(self._log_path / experiment_name)

    def log_at_startup(self) -> None:
        keys_to_drop = [
            "_end_queue",
            "_queue_config",
            "_status_queue",
            "_storage",
            "rng",
            "run_paths",
            "substitutions",
        ]
        settings_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in keys_to_drop
        }
        logger.info(f"Running '{self.name()}' with settings {settings_dict}")

    @classmethod
    @abstractmethod
    def name(cls) -> str: ...

    @classmethod
    def display_name(cls) -> str:
        return cls.name()

    @classmethod
    @abstractmethod
    def description(cls) -> str: ...

    @classmethod
    def group(cls) -> str | None:
        """Default value to prevent errors in children classes
        since only EnsembleExperiment and EnsembleSmoother should
        override it
        """
        return None

    def send_event(self, event: StatusEvents) -> None:
        self._status_queue.put(event)

    def send_smoother_event(
        self, iteration: int, run_id: uuid.UUID, event: AnalysisEvent
    ) -> None:
        match event:
            case AnalysisStatusEvent(msg=msg):
                self.send_event(
                    RunModelStatusEvent(iteration=iteration, run_id=run_id, msg=msg)
                )
            case AnalysisTimeEvent():
                self.send_event(
                    RunModelTimeEvent(
                        iteration=iteration,
                        run_id=run_id,
                        elapsed_time=event.elapsed_time,
                        remaining_time=event.remaining_time,
                    )
                )
            case AnalysisErrorEvent():
                self.send_event(
                    RunModelErrorEvent(
                        iteration=iteration,
                        run_id=run_id,
                        error_msg=event.error_msg,
                        data=event.data,
                    )
                )
            case AnalysisDataEvent(name=name, data=data):
                self.send_event(
                    RunModelDataEvent(
                        iteration=iteration, run_id=run_id, name=name, data=data
                    )
                )
            case AnalysisCompleteEvent():
                self.send_event(
                    RunModelUpdateEndEvent(
                        iteration=iteration, run_id=run_id, data=event.data
                    )
                )

    @property
    def queue_system(self) -> QueueSystem:
        return self._queue_config.queue_system

    @property
    def ensemble_size(self) -> int:
        return len(self._initial_realizations_mask)

    def cancel(self) -> None:
        self._end_queue.put("END")

    def has_failed_realizations(self) -> bool:
        return any(self._create_mask_from_failed_realizations())

    def _create_mask_from_failed_realizations(self) -> list[bool]:
        """
        Creates a list of bools representing the failed realizations,
        i.e., a realization that has failed is assigned a True value.
        """
        if self._completed_realizations_mask:
            return [
                initial and not completed
                for initial, completed in zip(
                    self._initial_realizations_mask,
                    self._completed_realizations_mask,
                    strict=False,
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
        self._context_env[key] = value
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
        for key in list(self._context_env.keys()):
            self._context_env.pop(key)
            os.environ.pop(key, None)

    @tracer.start_as_current_span(f"{__name__}.start_simulations_thread")
    def start_simulations_thread(
        self,
        evaluator_server_config: EvaluatorServerConfig,
        restart: bool = False,
    ) -> None:
        failed = False
        exception: Exception | None = None
        error_messages: MutableSequence[str] = []
        try:
            self.start_time = int(time.time())
            self.stop_time = None
            with captured_logs(error_messages):
                self._set_default_env_context()
                self.run_experiment(
                    evaluator_server_config=evaluator_server_config,
                    restart=restart,
                )
                if self._completed_realizations_mask:
                    combined = np.logical_or(
                        np.array(self._completed_realizations_mask),
                        np.array(self.active_realizations),
                    )
                    self._completed_realizations_mask = list(combined)
                else:
                    self._completed_realizations_mask = copy.copy(
                        self.active_realizations
                    )
        except ErtRunError as e:
            self._completed_realizations_mask = []
            failed = True
            exception = e
        except UserWarning:
            pass
        except Exception as e:
            failed = True
            exception = e
        finally:
            self._clean_env_context()
            self.stop_time = int(time.time())

            self.send_event(
                EndEvent(
                    failed=failed,
                    msg=(
                        self.format_error(exception, error_messages)
                        if failed
                        else "Experiment completed."
                    ),
                )
            )

    @abstractmethod
    def run_experiment(
        self,
        evaluator_server_config: EvaluatorServerConfig,
        restart: bool = False,
    ) -> None: ...

    @staticmethod
    def format_error(
        exception: Exception | None, error_messages: MutableSequence[str]
    ) -> str:
        msg = "\n".join(error_messages)
        if exception is None:
            return msg
        return f"{exception}\n{msg}"

    def get_runtime(self) -> int:
        if self.start_time is None:
            return 0
        elif self.stop_time is None:
            return round(time.time() - self.start_time)
        return self.stop_time - self.start_time

    def get_current_status(self) -> dict[str, int]:
        status: dict[str, int] = defaultdict(int)
        if self._iter_snapshot.keys():
            current_iter = max(list(self._iter_snapshot.keys()))
            all_realizations = self._iter_snapshot[current_iter].reals

            if all_realizations:
                for real in all_realizations.values():
                    status[str(real["status"])] += 1

        if self.restart:
            status["Finished"] += (
                self._get_number_of_finished_realizations_from_reruns()
            )
        return dict(status)

    def _get_number_of_finished_realizations_from_reruns(self) -> int:
        return self.active_realizations.count(
            False
        ) - self._initial_realizations_mask.count(False)

    def get_current_snapshot(self) -> EnsembleSnapshot:
        if self._iter_snapshot.keys():
            current_iter = max(list(self._iter_snapshot.keys()))
            return self._iter_snapshot[current_iter]
        return EnsembleSnapshot()

    def get_memory_consumption(self) -> int:
        max_memory_consumption: int = 0
        if self._iter_snapshot.keys():
            current_iter = max(list(self._iter_snapshot.keys()))
            for fm in self._iter_snapshot[current_iter].get_all_fm_steps().values():
                max_usage = fm.get("max_memory_usage", "0")
                if max_usage:
                    max_memory_consumption = max(int(max_usage), max_memory_consumption)

        return max_memory_consumption

    def _current_progress(self) -> tuple[float, int]:
        current_iter = max(list(self._iter_snapshot.keys()))
        done_realizations = self.active_realizations.count(False)
        all_realizations = self._iter_snapshot[current_iter].reals
        current_progress = 0.0
        realization_count = self.get_number_of_active_realizations()

        if all_realizations:
            for real in all_realizations.values():
                if real["status"] in {
                    REALIZATION_STATE_FINISHED,
                    REALIZATION_STATE_FAILED,
                }:
                    done_realizations += 1

            realization_progress = float(done_realizations) / len(
                self.active_realizations
            )
            current_progress = (
                (current_iter + realization_progress) / self._total_iterations
                if self._total_iterations != 1
                else realization_progress
            )

        return current_progress, realization_count

    def send_snapshot_event(self, event: Event, iteration: int) -> None:
        if type(event) is EESnapshot:
            snapshot = EnsembleSnapshot.from_nested_dict(event.snapshot)
            self._iter_snapshot[iteration] = snapshot
            current_progress, realization_count = self._current_progress()
            status = self.get_current_status()
            self.send_event(
                FullSnapshotEvent(
                    iteration_label=f"Running forecast for iteration: {iteration}",
                    total_iterations=self._total_iterations,
                    progress=current_progress,
                    realization_count=realization_count,
                    status_count=status,
                    iteration=iteration,
                    snapshot=copy.deepcopy(snapshot),
                )
            )
        elif type(event) is EESnapshotUpdate:
            if iteration not in self._iter_snapshot:
                raise OutOfOrderSnapshotUpdateException(
                    f"got snapshot update message without having stored "
                    f"snapshot for iter {iteration}"
                )
            snapshot = EnsembleSnapshot()
            snapshot.update_from_event(
                event, source_snapshot=self._iter_snapshot[iteration]
            )
            self._iter_snapshot[iteration].merge_snapshot(snapshot)
            current_progress, realization_count = self._current_progress()
            status = self.get_current_status()
            self.send_event(
                SnapshotUpdateEvent(
                    iteration_label=f"Running forecast for iteration: {iteration}",
                    total_iterations=self._total_iterations,
                    progress=current_progress,
                    realization_count=realization_count,
                    status_count=status,
                    iteration=iteration,
                    snapshot=copy.deepcopy(snapshot),
                )
            )

    async def run_monitor(
        self, ee_config: EvaluatorServerConfig, iteration: int
    ) -> bool:
        try:
            logger.debug("connecting to new monitor...")
            async with Monitor(ee_config.get_uri(), ee_config.token) as monitor:
                logger.debug("connected")
                async for event in monitor.track(heartbeat_interval=0.1):
                    if type(event) in {
                        EESnapshot,
                        EESnapshotUpdate,
                    }:
                        event = cast(EESnapshot | EESnapshotUpdate, event)

                        self.send_snapshot_event(event, iteration)

                        if event.snapshot.get(STATUS) in {
                            ENSEMBLE_STATE_STOPPED,
                            ENSEMBLE_STATE_FAILED,
                        }:
                            logger.debug(
                                "observed evaluation stopped event, signal done"
                            )
                            await monitor.signal_done()

                        if event.snapshot.get(STATUS) == ENSEMBLE_STATE_CANCELLED:
                            logger.debug(
                                "observed evaluation cancelled event, exit drainer"
                            )
                            raise UserCancelled(
                                "Experiment cancelled by user during evaluation"
                            )
                    elif type(event) is EETerminated:
                        logger.debug("got terminated event")

                    if not self._end_queue.empty():
                        logger.debug("Run model canceled - during evaluation")
                        self._end_queue.get()
                        await monitor.signal_cancel()
                        logger.debug(
                            "Run model canceled - during evaluation - cancel sent"
                        )
        except UserCancelled:
            raise
        except Exception as e:
            logger.exception(f"unexpected error: {e}")
            # We really don't know what happened...  shut down
            # the thread and get out of here. The monitor has
            # been stopped by the ctx-mgr
            return False

        return True

    async def run_ensemble_evaluator_async(
        self,
        run_args: list[RunArg],
        ensemble: Ensemble,
        ee_config: EvaluatorServerConfig,
    ) -> list[int]:
        if not self._end_queue.empty():
            logger.debug("Run model canceled - pre evaluation")
            self._end_queue.get()
            raise UserCancelled("Experiment cancelled by user in pre evaluation")

        ee_ensemble = self._build_ensemble(run_args, ensemble.experiment_id)
        evaluator = EnsembleEvaluator(
            ee_ensemble,
            ee_config,
        )
        evaluator_task = asyncio.create_task(
            evaluator.run_and_get_successful_realizations()
        )
        await evaluator._server_started
        if not (await self.run_monitor(ee_config, ensemble.iteration)):
            await evaluator_task
            return []

        logger.debug("observed that model was finished, waiting tasks completion...")
        # The model has finished, we indicate this by sending a DONE
        logger.debug("tasks complete")

        if not self._end_queue.empty():
            logger.debug("Run model canceled - post evaluation")
            self._end_queue.get()
            try:
                await evaluator_task
            except Exception as e:
                raise Exception(
                    "Exception occured during user initiatied termination of experiment"
                ) from e
            raise UserCancelled("Experiment cancelled by user in post evaluation")

        await evaluator_task
        ensemble.refresh_ensemble_state()

        return evaluator_task.result()

    # This function needs to be there for the sake of testing that expects sync ee run
    @tracer.start_as_current_span(f"{__name__}.run_ensemble_evaluator")
    def run_ensemble_evaluator(
        self,
        run_args: list[RunArg],
        ensemble: Ensemble,
        ee_config: EvaluatorServerConfig,
    ) -> list[int]:
        return asyncio.run(
            self.run_ensemble_evaluator_async(run_args, ensemble, ee_config)
        )

    def _build_ensemble(
        self,
        run_args: list[RunArg],
        experiment_id: uuid.UUID,
    ) -> EEEnsemble:
        realizations = []
        for run_arg in run_args:
            realizations.append(
                Realization(
                    active=run_arg.active,
                    iens=run_arg.iens,
                    fm_steps=self._forward_model_steps,
                    max_runtime=self._queue_config.max_runtime,
                    run_arg=run_arg,
                    num_cpu=self._queue_config.preferred_num_cpu,
                    job_script=self._queue_config.job_script,
                    realization_memory=self._queue_config.realization_memory,
                )
            )
        return EEEnsemble(
            realizations,
            {},
            self._queue_config,
            self.minimum_required_realizations,
            str(experiment_id),
        )

    @property
    def paths(self) -> list[str]:
        run_paths = []
        active_realizations = np.where(self.active_realizations)[0]
        for iteration in range(
            self.start_iteration, self._total_iterations + self.start_iteration
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
        realization_set = {Path(run_path).parent for run_path in self.paths}
        return [real_path.exists() for real_path in realization_set].count(True)

    def get_number_of_active_realizations(self) -> int:
        return (
            self._initial_realizations_mask.count(True)
            if self.restart
            else self.active_realizations.count(True)
        )

    def get_number_of_successful_realizations(self) -> int:
        return self.active_realizations.count(True)

    @log_duration(logger, logging.INFO)
    def rm_run_path(self) -> None:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(delete_runpath, self.paths)

    def validate_successful_realizations_count(self) -> None:
        successful_realizations_count = self.get_number_of_successful_realizations()
        min_realization_count = self.minimum_required_realizations

        if successful_realizations_count < min_realization_count:
            raise ValueError(
                f"Number of successful realizations ({successful_realizations_count}) is less "
                f"than the specified MIN_REALIZATIONS"
                f"({min_realization_count})"
            )

    @tracer.start_as_current_span(f"{__name__}.run_workflows")
    def run_workflows(
        self,
        runtime: HookRuntime,
        fixtures: WorkflowFixtures,
    ) -> None:
        for workflow in self._hooked_workflows[runtime]:
            WorkflowRunner(workflow=workflow, fixtures=fixtures).run_blocking()

    def _evaluate_and_postprocess(
        self,
        run_args: list[RunArg],
        ensemble: Ensemble,
        evaluator_server_config: EvaluatorServerConfig,
    ) -> int:
        create_run_path(
            run_args=run_args,
            ensemble=ensemble,
            user_config_file=str(self._user_config_file),
            env_vars=self._env_vars,
            env_pr_fm_step=self._env_pr_fm_step,
            forward_model_steps=self._forward_model_steps,
            substitutions=self._substitutions,
            templates=self._templates,
            model_config=self._model_config,
            runpaths=self.run_paths,
            context_env=self._context_env,
        )

        self.run_workflows(
            HookRuntime.PRE_SIMULATION,
            fixtures={
                "storage": self._storage,
                "ensemble": ensemble,
                "reports_dir": self.reports_dir(
                    experiment_name=ensemble.experiment.name
                ),
                "random_seed": self.random_seed,
                "run_paths": self.run_paths,
            },
        )
        try:
            successful_realizations = self.run_ensemble_evaluator(
                run_args,
                ensemble,
                evaluator_server_config,
            )
        except UserCancelled:
            self.active_realizations = [False for _ in self.active_realizations]
            raise

        starting_realizations = [real.iens for real in run_args if real.active]
        failed_realizations = list(
            set(starting_realizations) - set(successful_realizations)
        )
        for iens in failed_realizations:
            self.active_realizations[iens] = False

        num_successful_realizations = len(successful_realizations)
        self.validate_successful_realizations_count()
        logger.info(f"Experiment ran on QUEUESYSTEM: {self._queue_config.queue_system}")
        logger.info(f"Experiment ran with number of realizations: {self.ensemble_size}")
        logger.info(
            f"Experiment run ended with number of realizations succeeding: "
            f"{num_successful_realizations}"
        )
        logger.info(
            f"Experiment run ended with number of realizations failing: "
            f"{self.ensemble_size - num_successful_realizations}"
        )
        logger.info(f"Experiment run finished in: {self.get_runtime()}s")
        self.run_workflows(
            HookRuntime.POST_SIMULATION,
            fixtures={
                "storage": self._storage,
                "ensemble": ensemble,
                "reports_dir": self.reports_dir(
                    experiment_name=ensemble.experiment.name
                ),
                "random_seed": self.random_seed,
                "run_paths": self.run_paths,
            },
        )

        return num_successful_realizations


class UpdateRunModel(BaseRunModel):
    def __init__(
        self,
        analysis_settings: ESSettings,
        update_settings: UpdateSettings,
        storage: Storage,
        runpath_file: Path,
        user_config_file: Path,
        env_vars: dict[str, str],
        env_pr_fm_step: dict[str, dict[str, Any]],
        model_config: ModelConfig,
        queue_config: QueueConfig,
        forward_model_steps: list[ForwardModelStep],
        status_queue: SimpleQueue[StatusEvents],
        substitutions: Substitutions,
        templates: list[tuple[str, str]],
        hooked_workflows: defaultdict[HookRuntime, list[Workflow]],
        active_realizations: list[bool],
        total_iterations: int,
        start_iteration: int,
        random_seed: int | None,
        minimum_required_realizations: int,
        log_path: Path,
    ):
        self._analysis_settings: ESSettings = analysis_settings
        self._update_settings: UpdateSettings = update_settings

        super().__init__(
            storage,
            runpath_file,
            user_config_file,
            env_vars,
            env_pr_fm_step,
            model_config,
            queue_config,
            forward_model_steps,
            status_queue,
            substitutions,
            templates,
            hooked_workflows,
            active_realizations=active_realizations,
            total_iterations=total_iterations,
            start_iteration=start_iteration,
            random_seed=random_seed,
            minimum_required_realizations=minimum_required_realizations,
            log_path=log_path,
        )

    def update(
        self, prior: Ensemble, posterior_name: str, weight: float = 1.0
    ) -> Ensemble:
        self.validate_successful_realizations_count()
        self.send_event(
            RunModelUpdateBeginEvent(iteration=prior.iteration, run_id=prior.id)
        )
        self.send_event(
            RunModelStatusEvent(
                iteration=prior.iteration,
                run_id=prior.id,
                msg="Creating posterior ensemble..",
            )
        )

        workflow_fixtures: WorkflowFixtures = {
            "storage": self._storage,
            "ensemble": prior,
            "observation_settings": self._update_settings,
            "es_settings": self._analysis_settings,
            "random_seed": self.random_seed,
            "reports_dir": self.reports_dir(experiment_name=prior.experiment.name),
            "run_paths": self.run_paths,
        }

        posterior = self._storage.create_ensemble(
            prior.experiment,
            ensemble_size=prior.ensemble_size,
            iteration=prior.iteration + 1,
            name=posterior_name,
            prior_ensemble=prior,
        )
        if prior.iteration == 0:
            self.run_workflows(
                HookRuntime.PRE_FIRST_UPDATE,
                fixtures=workflow_fixtures,
            )
        self.run_workflows(
            HookRuntime.PRE_UPDATE,
            fixtures=workflow_fixtures,
        )
        try:
            smoother_update(
                prior,
                posterior,
                update_settings=self._update_settings,
                es_settings=self._analysis_settings,
                parameters=prior.experiment.update_parameters,
                observations=prior.experiment.observation_keys,
                global_scaling=weight,
                rng=self.rng,
                progress_callback=functools.partial(
                    self.send_smoother_event,
                    prior.iteration,
                    prior.id,
                ),
            )
        except ErtAnalysisError as e:
            raise ErtRunError(
                "Update algorithm failed for iteration:"
                f"{posterior.iteration}. The following error occurred: {e}"
            ) from e
        self.run_workflows(
            HookRuntime.POST_UPDATE,
            fixtures=workflow_fixtures,
        )
        return posterior
