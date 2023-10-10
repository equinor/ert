from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Union

from ert.cli import MODULE_MODE
from ert.config import HookRuntime, QueueSystem
from ert.enkf_main import EnKFMain
from ert.ensemble_evaluator import (
    Ensemble,
    EnsembleBuilder,
    EnsembleEvaluator,
    EvaluatorServerConfig,
    LegacyJob,
    LegacyStep,
    RealizationBuilder,
)
from ert.job_queue import RunStatus
from ert.libres_facade import LibresFacade
from ert.run_context import RunContext
from ert.storage import StorageAccessor

event_logger = logging.getLogger("ert.event_log")

if TYPE_CHECKING:
    from ert.config import QueueConfig
    from ert.run_models.run_arguments import RunArgumentsType


class ErtRunError(Exception):
    pass


class _LogAggregration(logging.Handler):
    def __init__(self) -> None:
        self.messages: List[str] = []
        self.exclude_logs = ["opencensus.ext.azure.common.transport"]
        super().__init__()

    def emit(self, record: logging.LogRecord) -> None:
        if record.name not in self.exclude_logs:
            self.messages.append(record.getMessage())


@contextmanager
def captured_logs(level: int = logging.ERROR) -> Iterator[_LogAggregration]:
    handler = _LogAggregration()
    root_logger = logging.getLogger()
    handler.setLevel(level)
    root_logger.addHandler(handler)
    try:
        yield handler
    finally:
        root_logger.removeHandler(handler)


# pylint: disable=too-many-instance-attributes,too-many-public-methods
# pylint: disable=too-many-arguments
class BaseRunModel:
    def __init__(
        self,
        simulation_arguments: RunArgumentsType,
        ert: EnKFMain,
        facade: LibresFacade,
        storage: StorageAccessor,
        queue_config: QueueConfig,
        experiment_id: uuid.UUID,
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

        self._job_start_time: int = 0
        self._job_stop_time: int = 0
        self._indeterminate: bool = False
        self._fail_message: str = ""
        self._failed: bool = False
        self._queue_config: QueueConfig = queue_config
        self._initial_realizations_mask: List[bool] = []
        self._completed_realizations_mask: List[bool] = []
        self.support_restart: bool = True
        self.facade = facade
        self.ert = ert
        self._storage = storage
        self._simulation_arguments = simulation_arguments
        self._experiment_id = experiment_id
        self.reset()
        # mapping from iteration number to ensemble id
        self._iter_map: Dict[int, str] = {}
        self.validate()
        self._context_env_keys: List[str] = []

    @property
    def queue_system(self) -> QueueSystem:
        return self._queue_config.queue_system

    @property
    def simulation_arguments(self) -> RunArgumentsType:
        return self._simulation_arguments

    @property
    def _ensemble_size(self) -> int:
        return len(self._initial_realizations_mask)

    @property
    def _active_realizations(self) -> List[int]:
        return [
            idx
            for idx, mask_val in enumerate(self._initial_realizations_mask)
            if mask_val
        ]

    def reset(self) -> None:
        self._failed = False
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
        self.set_env_key("_ERT_EXPERIMENT_ID", str(self._experiment_id))

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
        self.startSimulations(
            evaluator_server_config=evaluator_server_config,
        )

    def startSimulations(self, evaluator_server_config: EvaluatorServerConfig) -> None:
        logs: _LogAggregration = _LogAggregration()
        try:
            with captured_logs() as logs:
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
            self._fail_message = str(e) + "\n" + "\n".join(sorted(logs.messages))
            self._simulationEnded()
        except UserWarning as e:
            self._fail_message = str(e) + "\n" + "\n".join(sorted(logs.messages))
            self._simulationEnded()
        except Exception as e:
            self._failed = True
            self._fail_message = str(e) + "\n" + "\n".join(sorted(logs.messages))
            self._simulationEnded()
            raise

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
        return self._fail_message

    def _simulationEnded(self) -> None:
        self._clean_env_context()
        self._job_stop_time = int(time.time())

    def setPhase(
        self, phase: int, phase_name: str, indeterminate: Optional[bool] = None
    ) -> None:
        self.setPhaseName(phase_name)
        if not 0 <= phase <= self._phase_count:
            raise ValueError(
                f"Phase must be integer between (inclusive) 0 and {self._phase_count}"
            )

        self.setIndeterminate(indeterminate)

        if phase == 0:
            self._job_start_time = int(time.time())

        if phase == self._phase_count:
            self._simulationEnded()

        self._phase = phase

    def stop_time(self) -> int:
        return self._job_stop_time

    def start_time(self) -> int:
        return self._job_start_time

    def get_runtime(self) -> Union[int, float]:
        if self.stop_time() < self.start_time():
            return time.time() - self.start_time()
        else:
            return self.stop_time() - self.start_time()

    def isIndeterminate(self) -> bool:
        return not self.isFinished() and self._indeterminate

    def checkHaveSufficientRealizations(self, num_successful_realizations: int) -> None:
        if num_successful_realizations == 0:
            raise ErtRunError("Experiment failed! All realizations failed!")
        if not self.ert.analysisConfig().have_enough_realisations(
            num_successful_realizations
        ):
            raise ErtRunError(
                "Too many realizations have failed! "
                f"Number of successful realizations: {num_successful_realizations}, "
                "number of active realizations: "
                f"{self._simulation_arguments.active_realizations.count(True)}, "
                "expected minimal number of successful realizations: "
                f"{self.ert.analysisConfig().minimum_required_realizations}\n"
                "You can add/adjust MIN_REALIZATIONS "
                "to allow (more) failures in your experiments."
            )

    def run_ensemble_evaluator(
        self, run_context: RunContext, ee_config: EvaluatorServerConfig
    ) -> int:
        ensemble = self._build_ensemble(run_context)

        totalOk = EnsembleEvaluator(
            ensemble,
            ee_config,
            run_context.iteration,
        ).run_and_get_successful_realizations()

        self.deactivate_failed_jobs(run_context)

        run_context.sim_fs.sync()
        return totalOk

    @staticmethod
    def deactivate_failed_jobs(run_context: RunContext) -> None:
        for iens, run_arg in enumerate(run_context):
            if run_context.is_active(iens) and run_arg.run_status in (
                RunStatus.JOB_LOAD_FAILURE,
                RunStatus.JOB_RUN_FAILURE,
            ):
                run_context.deactivate_realization(iens)

    def _build_ensemble(
        self,
        run_context: RunContext,
    ) -> "Ensemble":
        builder = EnsembleBuilder().set_legacy_dependencies(
            self._queue_config,
            self.ert.analysisConfig(),
        )

        for iens, run_arg in enumerate(run_context):
            active = run_context.is_active(iens)
            real = RealizationBuilder().set_iens(iens).active(active)
            if active:
                jobs = [
                    LegacyJob(
                        id_=str(index),
                        index=str(index),
                        name=ext_job.name,
                        ext_job=ext_job,
                    )
                    for index, ext_job in enumerate(
                        self.ert.resConfig().forward_model_list
                    )
                ]
                real.add_step(
                    LegacyStep(
                        id_="0",
                        jobs=jobs,
                        name="legacy step",
                        max_runtime=self.ert.analysisConfig().max_runtime,
                        run_arg=run_arg,
                        num_cpu=self.ert.get_num_cpu(),
                        job_script=self.ert.resConfig().queue_config.job_script,
                    )
                )
            builder.add_realization(real)
        return builder.set_id(str(uuid.uuid1()).split("-", maxsplit=1)[0]).build()

    @property
    def id(self) -> uuid.UUID:
        return self._experiment_id

    def check_if_runpath_exists(self) -> bool:
        """
        Determine if the run_path exists by checking if it contains
        at least one iteration directory for the realizations in the active mask.
        The run_path can contain one or two %d specifiers ie:
            "realization-%d/iter-%d/"
            "realization-%d/"
        """
        start_iteration = self._simulation_arguments.start_iteration
        number_of_iterations = self.facade.number_of_iterations
        active_mask = self._simulation_arguments.active_realizations
        active_realizations = [i for i in range(len(active_mask)) if active_mask[i]]
        for iteration in range(start_iteration, number_of_iterations):
            run_paths = self.facade.get_run_paths(active_realizations, iteration)
            for run_path in run_paths:
                if Path(run_path).exists():
                    return True
        return False

    # pylint: disable=too-many-branches
    def validate(self) -> None:
        if self._simulation_arguments is None:
            return
        errors = []

        active_mask = self._simulation_arguments.active_realizations
        active_realizations_count = len(
            [i for i in range(len(active_mask)) if active_mask[i]]
        )

        min_realization_count: int = 0

        if self.ert:
            min_realization_count = (
                self.ert.analysisConfig().minimum_required_realizations
            )

        if (
            "SingleTestRun" not in str(type(self))
            and active_realizations_count < min_realization_count
        ):
            raise ValueError(
                f"Number of active realizations ({active_realizations_count}) is less "
                f"than the specified MIN_REALIZATIONS in the config file "
                f"({min_realization_count})"
            )

        current_case = self._simulation_arguments.current_case
        target_case = self._simulation_arguments.target_case

        if current_case is not None:
            try:
                case = self._storage.get_ensemble_by_name(current_case)
                if case.ensemble_size != self.ert.getEnsembleSize():
                    errors.append(
                        f"- Existing case: {current_case} was created with ensemble "
                        f"size smaller than specified in the ert configuration file ("
                        f"{case.ensemble_size} "
                        f" < {self.ert.getEnsembleSize()})"
                    )
            except KeyError:
                pass
        if target_case is not None:
            if target_case == current_case:
                errors.append(
                    f"- Target case and current case can not have the same name. "
                    f"They were both: {current_case}"
                )

            if "%d" in target_case:
                num_iterations = self._simulation_arguments.num_iterations
                for i in range(num_iterations):
                    try:
                        self._storage.get_ensemble_by_name(
                            target_case % i  # noqa: S001
                        )
                        errors.append(
                            f"- Target case: {target_case % i} exists"  # noqa: S001
                        )
                    except KeyError:
                        pass
            else:
                try:
                    self._storage.get_ensemble_by_name(target_case)
                    errors.append(f"- Target case: {target_case} exists")
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
        self.ert.createRunPath(run_context)

        phase_string = f"Pre processing for iteration: {iteration}"
        self.setPhaseName(phase_string, indeterminate=True)
        self.ert.runWorkflows(
            HookRuntime.PRE_SIMULATION, self._storage, run_context.sim_fs
        )

        phase_string = f"Running forecast for iteration: {iteration}"
        self.setPhaseName(phase_string, indeterminate=False)

        num_successful_realizations = self.run_ensemble_evaluator(
            run_context, evaluator_server_config
        )

        num_successful_realizations += (
            self._simulation_arguments.prev_successful_realizations
        )
        self.checkHaveSufficientRealizations(num_successful_realizations)

        phase_string = f"Post processing for iteration: {iteration}"
        self.setPhaseName(phase_string, indeterminate=True)
        self.ert.runWorkflows(
            HookRuntime.POST_SIMULATION, self._storage, run_context.sim_fs
        )

        return num_successful_realizations
