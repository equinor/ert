from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import dataclasses
import functools
import logging
import os
import queue
import shutil
import threading
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Generator, MutableSequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, Protocol

import numpy as np
from pydantic import BaseModel, PrivateAttr

from _ert.events import EEEvent, EESnapshot, EESnapshotUpdate
from ert.config import (
    ConfigValidationError,
    DesignMatrix,
    ForwardModelStep,
    GenKwConfig,
    HookRuntime,
    ModelConfig,
    ParameterConfig,
    QueueConfig,
    QueueSystem,
    Workflow,
)
from ert.enkf_main import create_run_path
from ert.ensemble_evaluator import Ensemble as EEEnsemble
from ert.ensemble_evaluator import (
    EnsembleEvaluator,
    EvaluatorServerConfig,
    Realization,
    WarningEvent,
)
from ert.ensemble_evaluator.evaluator import UserCancelled
from ert.ensemble_evaluator.snapshot import EnsembleSnapshot
from ert.ensemble_evaluator.state import (
    REALIZATION_STATE_FAILED,
    REALIZATION_STATE_FINISHED,
)
from ert.mode_definitions import MODULE_MODE
from ert.plugins import (
    HookedWorkflowFixtures,
    PostSimulationFixtures,
    PreSimulationFixtures,
)
from ert.runpaths import Runpaths
from ert.storage import Ensemble, Storage, open_storage
from ert.trace import tracer
from ert.utils import log_duration
from ert.warnings import PostSimulationWarning, capture_specific_warning
from ert.workflow_runner import WorkflowRunner

from ..plugins.workflow_fixtures import create_workflow_fixtures_from_hooked
from ..run_arg import RunArg
from .event import EndEvent, FullSnapshotEvent, SnapshotUpdateEvent, StatusEvents

logger = logging.getLogger(__name__)


class OutOfOrderSnapshotUpdateException(ValueError):
    pass


class ErtRunError(Exception):
    pass


class TooFewRealizationsSucceeded(ErtRunError):
    def __init__(
        self, successful_realizations: int, required_realizations: int
    ) -> None:
        self.message = (
            f"Number of successful realizations ({successful_realizations}) is less "
            "than the specified MIN_REALIZATIONS"
            f"({required_realizations})"
        )
        super().__init__(self.message)


def delete_runpath(run_path: str) -> None:
    if os.path.exists(run_path):
        shutil.rmtree(run_path)


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
        self,
        evaluator_server_config: EvaluatorServerConfig,
        rerun_failed_realizations: bool = False,
    ) -> None: ...


@dataclasses.dataclass
class RunModelAPI:
    experiment_name: str
    supports_rerunning_failed_realizations: bool
    start_simulations_thread: StartSimulationsThreadFn
    cancel: Callable[[], None]
    get_runtime: Callable[[], int]
    has_failed_realizations: Callable[[], bool]


class RunModel(BaseModel, ABC):
    storage_path: str
    runpath_file: Path
    user_config_file: Path
    env_vars: dict[str, str]
    env_pr_fm_step: dict[str, dict[str, Any]]
    runpath_config: ModelConfig
    queue_config: QueueConfig
    forward_model_steps: list[ForwardModelStep]
    substitutions: dict[str, str]
    hooked_workflows: defaultdict[HookRuntime, list[Workflow]]
    active_realizations: list[bool]
    log_path: Path
    random_seed: int
    start_iteration: int = 0
    minimum_required_realizations: int = 0
    supports_rerunning_failed_realizations: ClassVar[bool] = False

    # Private attributes initialized in model_post_init
    _start_time: int | None = PrivateAttr(None)
    _stop_time: int | None = PrivateAttr(None)
    _initial_realizations_mask: list[bool] = PrivateAttr()
    _completed_realizations_mask: list[bool] = PrivateAttr(default_factory=list)
    _storage: Storage = PrivateAttr()
    _context_env: dict[str, str] = PrivateAttr(default_factory=dict)
    _model_config: ModelConfig = PrivateAttr()
    _rng: np.random.Generator = PrivateAttr()
    _end_event: threading.Event = PrivateAttr(default_factory=threading.Event)
    _iter_snapshot: dict[int, EnsembleSnapshot] = PrivateAttr(default_factory=dict)
    _is_rerunning_failed_realizations: bool = PrivateAttr(False)
    _run_paths: Runpaths = PrivateAttr()
    _total_iterations: int = PrivateAttr(default=1)

    def __init__(
        self,
        *,
        status_queue: queue.SimpleQueue[StatusEvents],
        _total_iterations: int | None = None,
        **data: Any,
    ) -> None:
        super().__init__(**data)
        self._status_queue = status_queue

        if _total_iterations is not None:
            self._total_iterations = _total_iterations

    def model_post_init(self, ctx: Any) -> None:
        self._initial_realizations_mask = self.active_realizations.copy()
        self._completed_realizations_mask = [False] * len(self.active_realizations)
        self._storage = open_storage(self.storage_path, mode="w")
        self._rng = np.random.default_rng(self.random_seed)
        self._model_config = self.runpath_config

        self._run_paths = Runpaths(
            jobname_format=self._model_config.jobname_format_string,
            runpath_format=self._model_config.runpath_format_string,
            filename=str(self.runpath_file),
            substitutions=self.substitutions,
            eclbase=self._model_config.eclbase_format_string,
        )

    @property
    def api(self) -> RunModelAPI:
        return RunModelAPI(
            experiment_name=self.name(),
            get_runtime=self.get_runtime,
            start_simulations_thread=self.start_simulations_thread,
            has_failed_realizations=self.has_failed_realizations,
            supports_rerunning_failed_realizations=self.supports_rerunning_failed_realizations,
            cancel=self.cancel,
        )

    def reports_dir(self, experiment_name: str) -> str:
        return str(self.log_path / experiment_name)

    def log_at_startup(self) -> None:
        keys_to_drop = [
            "_end_event",
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
        settings_summary = {
            "run_model": self.name(),
            "num_realizations": self.runpath_config.num_realizations,
            "num_active_realizations": self.active_realizations.count(True),
            "num_parameters": sum(
                len(param_config.parameter_keys)
                for param_config in self.parameter_configuration
            )
            if hasattr(self, "parameter_configuration")
            else "NA",
            "localization": getattr(
                settings_dict.get("analysis_settings", {}), "localization", "NA"
            ),
        }

        logger.info(
            f"Running '{self.name()}'\n\n"
            f"Settings summary: {settings_summary}\n\n"
            f"Settings: {settings_dict}"
        )

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

    @property
    def queue_system(self) -> QueueSystem:
        return self.queue_config.queue_system

    @property
    def ensemble_size(self) -> int:
        return len(self._initial_realizations_mask)

    def cancel(self) -> None:
        self._end_event.set()

    def has_failed_realizations(self) -> bool:
        return any(self._create_mask_from_failed_realizations())

    def _create_mask_from_failed_realizations(self) -> list[bool]:
        """
        Creates a list of bools representing the failed realizations,
        i.e., a realization that has failed is assigned a True value.
        """
        return [
            initial and not completed
            for initial, completed in zip(
                self._initial_realizations_mask,
                self._completed_realizations_mask,
                strict=False,
            )
        ]

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
        rerun_failed_realizations: bool = False,
    ) -> None:
        failed = False
        exception: Exception | None = None
        error_messages: MutableSequence[str] = []
        traceback_str: str | None = None

        def handle_captured_event(message: Warning | str) -> None:
            self.send_event(WarningEvent(msg=str(message)))

        try:
            self._start_time = int(time.time())
            self._stop_time = None
            with (
                capture_specific_warning(PostSimulationWarning, handle_captured_event),
                captured_logs(error_messages),
            ):
                self._set_default_env_context()

                if (
                    rerun_failed_realizations
                    and not self.supports_rerunning_failed_realizations
                ):
                    raise ErtRunError(
                        f"Run model {self.name()} does not support "
                        f"restart/rerun of failed simulations."
                    )

                if rerun_failed_realizations:
                    self._storage = open_storage(self.storage_path, mode="w")
                    self.active_realizations = (
                        self._create_mask_from_failed_realizations()
                    )
                    logger.info(
                        f"Rerunning failed simulations for run model '{self.name()}'"
                    )

                self.run_experiment(
                    evaluator_server_config=evaluator_server_config,
                    rerun_failed_realizations=rerun_failed_realizations,
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
                self._storage.close()
        except ErtRunError as e:
            failed = True
            exception = e
        except UserWarning as e:
            logger.exception(e)
        except UserCancelled as e:
            failed = True
            exception = e
        except Exception as e:
            failed = True
            exception = e
            traceback_str = traceback.format_exc()
        finally:
            self._storage.close()
            self._clean_env_context()
            self._stop_time = int(time.time())
            self.send_event(
                EndEvent(
                    failed=failed,
                    msg=(
                        self.format_error(
                            exception=exception,
                            error_messages=error_messages,
                            traceback=traceback_str,
                        )
                        if failed
                        else "Experiment completed."
                    ),
                )
            )

    @abstractmethod
    def run_experiment(
        self,
        evaluator_server_config: EvaluatorServerConfig,
        rerun_failed_realizations: bool = False,
    ) -> None: ...

    @staticmethod
    def format_error(
        exception: Exception | None,
        error_messages: MutableSequence[str],
        traceback: str | None,
    ) -> str:
        msg = "\n".join(error_messages)
        if exception is None:
            return msg
        if traceback is None:
            return f"{exception}\n{msg}"
        return f"{exception}\n{traceback}\n{msg}"

    def get_runtime(self) -> int:
        if self._start_time is None:
            return 0
        elif self._stop_time is None:
            return round(time.time() - self._start_time)
        return self._stop_time - self._start_time

    def get_current_status(self) -> dict[str, int]:
        status: dict[str, int] = defaultdict(int)
        if self._iter_snapshot.keys():
            current_iter = max(list(self._iter_snapshot.keys()))
            all_realizations = self._iter_snapshot[current_iter].reals

            if all_realizations:
                for real in all_realizations.values():
                    status[str(real["status"])] += 1

        if self._is_rerunning_failed_realizations:
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

    def calculate_current_progress(self) -> float:
        current_iter = max(list(self._iter_snapshot.keys()))
        done_realizations = self.active_realizations.count(False)
        all_realizations = self._iter_snapshot[current_iter].reals
        current_progress = 0.0

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

            current_it_offset = current_iter - min(list(self._iter_snapshot.keys()))

            current_progress = (
                (current_it_offset + realization_progress) / self._total_iterations
                if self._total_iterations != 1
                else realization_progress
            )

        return current_progress

    def forward_event_from_ee(
        self,
        event: EEEvent,
        iteration: int,
    ) -> None:
        if type(event) is EESnapshot:
            snapshot = EnsembleSnapshot.from_nested_dict(event.snapshot)
            self._iter_snapshot[iteration] = snapshot
            current_progress = self.calculate_current_progress()
            realization_count = self.get_number_of_active_realizations()
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
            current_progress = self.calculate_current_progress()
            realization_count = self.get_number_of_active_realizations()
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

    async def run_ensemble_evaluator_async(
        self,
        run_args: list[RunArg],
        ensemble: Ensemble,
        ee_config: EvaluatorServerConfig,
    ) -> list[int]:
        if self._end_event.is_set():
            logger.debug("Run model cancelled - pre evaluation")
            raise UserCancelled("Experiment cancelled by user in pre evaluation")

        ee_ensemble = self._build_ensemble(run_args, ensemble.experiment_id)
        evaluator = EnsembleEvaluator(
            ee_ensemble,
            ee_config,
            end_event=self._end_event,
            event_handler=functools.partial(
                self.forward_event_from_ee, iteration=ensemble.iteration
            ),
        )
        evaluator_task = asyncio.create_task(
            evaluator.run_and_get_successful_realizations()
        )
        try:
            if (await evaluator.wait_for_evaluation_result()) is not True:
                await evaluator_task
                return []
        finally:
            await evaluator_task

        logger.debug("tasks complete")

        if self._end_event.is_set():
            logger.debug("Run model cancelled - post evaluation")
            try:
                await evaluator_task
            except Exception as e:
                raise Exception(
                    "Exception occured during user initiated termination of experiment"
                ) from e
            raise UserCancelled("Experiment cancelled by user in post evaluation")

        await evaluator_task
        try:
            ensemble.refresh_ensemble_state()
        except OSError as err:
            logger.error(f"Got OSError when refreshing ensemble state: {err}")

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
                    fm_steps=self.forward_model_steps,
                    max_runtime=self.queue_config.max_runtime,
                    run_arg=run_arg,
                    num_cpu=self.queue_config.queue_options.num_cpu,
                    job_script=self.queue_config.queue_options.job_script,
                    realization_memory=self.queue_config.queue_options.realization_memory,
                )
            )
        return EEEnsemble(
            realizations,
            {},
            self.queue_config,
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
            run_paths.extend(self._run_paths.get_paths(active_realizations, iteration))
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
            if self._is_rerunning_failed_realizations
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
            raise TooFewRealizationsSucceeded(
                successful_realizations_count, min_realization_count
            )

    @tracer.start_as_current_span(f"{__name__}.run_workflows")
    def run_workflows(
        self,
        fixtures: HookedWorkflowFixtures,
    ) -> None:
        for workflow in self.hooked_workflows[fixtures.hook]:
            WorkflowRunner(
                workflow=workflow,
                fixtures=create_workflow_fixtures_from_hooked(fixtures),
            ).run_blocking()

    def _evaluate_and_postprocess(
        self,
        run_args: list[RunArg],
        ensemble: Ensemble,
        evaluator_server_config: EvaluatorServerConfig,
    ) -> int:
        create_run_path(
            run_args=run_args,
            ensemble=ensemble,
            user_config_file=str(self.user_config_file),
            env_vars=self.env_vars,
            env_pr_fm_step=self.env_pr_fm_step,
            forward_model_steps=self.forward_model_steps,
            substitutions=self.substitutions,
            parameters_file=self._model_config.gen_kw_export_name,
            runpaths=self._run_paths,
            context_env=self._context_env,
        )

        self.run_workflows(
            fixtures=PreSimulationFixtures(
                storage=self._storage,
                ensemble=ensemble,
                reports_dir=self.reports_dir(experiment_name=ensemble.experiment.name),
                random_seed=self.random_seed,
                run_paths=self._run_paths,
            ),
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
        logger.info(f"Experiment ran on QUEUESYSTEM: {self.queue_config.queue_system}")
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
            fixtures=PostSimulationFixtures(
                storage=self._storage,
                ensemble=ensemble,
                reports_dir=self.reports_dir(experiment_name=ensemble.experiment.name),
                random_seed=self.random_seed,
                run_paths=self._run_paths,
            ),
        )

        return num_successful_realizations

    @classmethod
    def _merge_parameters_from_design_matrix(
        cls,
        parameters_config: list[ParameterConfig],
        design_matrix: DesignMatrix | None,
        rerun_failed_realizations: bool,
    ) -> tuple[list[ParameterConfig], DesignMatrix | None, GenKwConfig | None]:
        design_matrix_group = None
        # If a design matrix is present, we try to merge design matrix parameters
        # to the experiment parameters and set new active realizations
        if design_matrix is not None and not rerun_failed_realizations:
            try:
                parameters_config, design_matrix_group = (
                    design_matrix.merge_with_existing_parameters(parameters_config)
                )

            except ConfigValidationError as exc:
                raise ErtRunError(str(exc)) from exc

        return parameters_config, design_matrix, design_matrix_group
