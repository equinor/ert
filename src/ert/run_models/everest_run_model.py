from __future__ import annotations

import copy
import datetime
import functools
import json
import logging
import os
import queue
import random
import re
import shutil
import threading
import time
from pathlib import Path
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Type,
    TypedDict,
)

import seba_sqlite.sqlite_storage
from ropt.enums import EventType, OptimizerExitCode
from ropt.plan import BasicOptimizer, Event
from seba_sqlite import SqliteStorage

from ert.config import ErtConfig
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.storage import open_storage
from everest.config import EverestConfig
from everest.optimizer.everest2ropt import everest2ropt
from everest.simulator import Simulator
from everest.simulator.everest_to_ert import everest_to_ert_config
from everest.strings import EVEREST, SIMULATOR_END, SIMULATOR_START, SIMULATOR_UPDATE

from ..resources import all_shell_script_fm_steps
from .base_run_model import BaseRunModel, StatusEvents

if TYPE_CHECKING:
    from ert.simulator.batch_simulator_context import BatchContext, Status


# A number of settings for the table reporters:
RESULT_COLUMNS = {
    "result_id": "ID",
    "batch_id": "Batch",
    "functions.weighted_objective": "Total-Objective",
    "linear_constraints.violations": "IC-violation",
    "nonlinear_constraints.violations": "OC-violation",
    "functions.objectives": "Objective",
    "functions.constraints": "Constraint",
    "evaluations.variables": "Control",
    "linear_constraints.values": "IC-diff",
    "nonlinear_constraints.values": "OC-diff",
    "functions.scaled_objectives": "Scaled-Objective",
    "functions.scaled_constraints": "Scaled-Constraint",
    "evaluations.scaled_variables": "Scaled-Control",
    "nonlinear_constraints.scaled_values": "Scaled-OC-diff",
    "nonlinear_constraints.scaled_violations": "Scaled-OC-violation",
}
GRADIENT_COLUMNS = {
    "result_id": "ID",
    "batch_id": "Batch",
    "gradients.weighted_objective": "Total-Gradient",
    "gradients.objectives": "Grad-objective",
    "gradients.constraints": "Grad-constraint",
}
SIMULATION_COLUMNS = {
    "result_id": "ID",
    "batch_id": "Batch",
    "realization": "Realization",
    "evaluations.evaluation_ids": "Simulation",
    "evaluations.variables": "Control",
    "evaluations.objectives": "Objective",
    "evaluations.constraints": "Constraint",
    "evaluations.scaled_variables": "Scaled-Control",
    "evaluations.scaled_objectives": "Scaled-Objective",
    "evaluations.scaled_constraints": "Scaled-Constraint",
}
PERTURBATIONS_COLUMNS = {
    "result_id": "ID",
    "batch_id": "Batch",
    "realization": "Realization",
    "evaluations.perturbed_evaluation_ids": "Simulation",
    "evaluations.perturbed_variables": "Control",
    "evaluations.perturbed_objectives": "Objective",
    "evaluations.perturbed_constraints": "Constraint",
    "evaluations.scaled_perturbed_variables": "Scaled-Control",
    "evaluations.scaled_perturbed_objectives": "Scaled-Objective",
    "evaluations.scaled_perturbed_constraints": "Scaled-Constraint",
}
MIN_HEADER_LEN = 3

logger = logging.getLogger(__name__)


class SimulationStatus(TypedDict):
    status: Status
    progress: List[List[JobProgress]]
    batch_number: int


class JobProgress(TypedDict):
    name: str
    status: str
    error: Optional[str]
    start_time: Optional[datetime.datetime]
    end_time: Optional[datetime.datetime]
    realization: str
    simulation: str


class MonitorThreadErrorCallback(Protocol):
    def __call__(
        self,
        batch: int,
        simulation: Any,
        realization: str,
        fm_name: str,
        error_path: str,
    ) -> None: ...


class SimulationCallback(Protocol):
    def __call__(
        self, simulation_status: SimulationStatus | None, event: str
    ) -> str | None: ...


class _MonitorThread(threading.Thread):
    """Invoke a callback when a sim context status changes.
    This thread will run as long as the given context is running.
    If the status of the simulation context changes, the callback
    function will be called with the appropriate status change.
    Notice that there are two callbacks at play here.  We have one,
    EverestWorkflow._simulation_callback, which will notify us whenever a new
    simulation batch is starting, and the "user provided" callback,
    which will be called from this class whenever the status of the
    simulation context changes.
    """

    def __init__(
        self,
        context: BatchContext,
        callback: Optional[SimulationCallback] = None,
        error_callback: Optional[MonitorThreadErrorCallback] = None,
        delete_run_path: Optional[bool] = False,
        display_all_jobs: Optional[bool] = False,
    ) -> None:
        super(_MonitorThread, self).__init__()

        # temporarily living simulation context
        self._context: Optional[BatchContext] = context
        self._callback: SimulationCallback = (
            callback if callback is not None else lambda simulation_status, event: None
        )
        self._delete_run_path = delete_run_path
        self._display_all_jobs = display_all_jobs
        self._shutdown_flag = False  # used to gracefully shut down this thread
        self._error_callback = error_callback

    def _cleanup(self) -> None:
        # cleanup
        if self._delete_run_path and self._context is not None:
            for context_index in range(len(self._context)):
                if self._context.is_job_completed(context_index):
                    path_to_delete = self._context.run_path(context_index)
                    if os.path.isdir(path_to_delete):

                        def onerror(
                            _: Callable[..., Any],
                            path: str,
                            sys_info: tuple[
                                Type[BaseException], BaseException, TracebackType
                            ],
                        ) -> None:
                            logging.getLogger(EVEREST).debug(
                                "Failed to remove {}, {}".format(path, sys_info)
                            )

                        shutil.rmtree(path_to_delete, onerror=onerror)  # pylint: disable=deprecated-argument

        self._context = None
        self._callback = lambda *_, **__: None
        self._shutdown_flag = True

    @property
    def _batch_number(self) -> int:
        """
        Return the current batch number from context.
        """
        # Get the string name of current case
        assert self._context is not None
        batch_n_sim_string = self._context.get_ensemble().name

        search = re.search(r"batch_([0-9]+)", batch_n_sim_string)
        return int(search.groups()[-1]) if search is not None else -1

    def _simulation_status(self) -> SimulationStatus:
        assert self._context is not None

        def extract(path_str: str, key: str) -> str:
            regex = r"/{}_(\d+)/".format(key)
            found = next(re.finditer(regex, path_str), None)
            return found.group(1) if found is not None else "unknown"

        # if job is waiting, the status returned
        # by the job_progress() method is unreliable
        jobs_progress: List[List[JobProgress]] = []
        batch_number = self._batch_number
        for i in range(len(self._context)):
            progress_queue = self._context.job_progress(i)
            if self._context.is_job_waiting(i) or progress_queue is None:
                jobs_progress.append([])
            else:
                jobs: List[JobProgress] = []
                for fms in progress_queue.steps:
                    if (
                        not self._display_all_jobs
                        and fms.name in all_shell_script_fm_steps
                    ):
                        continue
                    realization = extract(fms.std_out_file, "geo_realization")
                    simulation = extract(fms.std_out_file, "simulation")
                    jobs.append(
                        {
                            "name": fms.name,
                            "status": fms.status,
                            "error": fms.error,
                            "start_time": fms.start_time,
                            "end_time": fms.end_time,
                            "realization": realization,
                            "simulation": simulation,
                        }
                    )
                    if fms.error is not None:
                        assert self._error_callback is not None
                        self._error_callback(
                            int(batch_number),
                            simulation,
                            realization,
                            fms.name,
                            fms.std_err_file,
                        )
                jobs_progress.append(jobs)
        return {
            "status": copy.deepcopy(self._context.status),
            "progress": jobs_progress,
            "batch_number": int(batch_number),
        }

    def run(self) -> None:
        if self._context is None:
            self._cleanup()
            return
        try:
            status = None
            while self._context.running() and not self._shutdown_flag:
                newstatus = self._simulation_status()
                if status == newstatus:  # No change in status
                    time.sleep(1)
                    continue
                signal = self._callback(
                    newstatus,
                    event=SIMULATOR_START if status is None else SIMULATOR_UPDATE,
                )
                status = newstatus
                if signal == "stop_queue":
                    self.stop()
            self._callback(status, event=SIMULATOR_END)
        finally:
            self._cleanup()

    def stop(self) -> None:
        if self._context is not None:
            self._context.stop()

        self._shutdown_flag = True


class OptimizerCallback(Protocol):
    def __call__(self) -> str | None: ...


class EverestRunModel(BaseRunModel):
    def __init__(
        self,
        config: ErtConfig,
        everest_config: EverestConfig,
        simulation_callback: SimulationCallback,
        optimization_callback: OptimizerCallback,
        display_all_jobs: bool = True,
    ):
        everest_config = self._add_defaults(everest_config)

        Path(everest_config.log_dir).mkdir(parents=True, exist_ok=True)
        Path(everest_config.optimization_output_dir).mkdir(parents=True, exist_ok=True)

        self.ropt_config = everest2ropt(everest_config)
        self.everest_config = everest_config
        self.support_restart = False

        self._sim_callback = simulation_callback
        self._opt_callback = optimization_callback
        self._monitor_thread: Optional[_MonitorThread] = None
        self._fm_errors: Dict[int, Dict[str, Any]] = {}
        self._simulation_delete_run_path = (
            False
            if everest_config.simulator is None
            else (everest_config.simulator.delete_run_path or False)
        )
        self._display_all_jobs = display_all_jobs
        self._result: Optional[seba_sqlite.sqlite_storage.OptimalResult] = None
        self._exit_code: Optional[
            Literal["max_batch_num_reached"] | OptimizerExitCode
        ] = None
        self._max_batch_num_reached = False

        storage = open_storage(config.ens_path, mode="w")
        status_queue: queue.SimpleQueue[StatusEvents] = queue.SimpleQueue()

        super().__init__(
            config,
            storage,
            config.queue_config,
            status_queue,
            active_realizations=[True] * config.model_config.num_realizations,
            minimum_required_realizations=config.model_config.num_realizations,  # OK?
        )

        self.num_retries_per_iter = 0  # OK?

    @staticmethod
    def _add_defaults(config: EverestConfig) -> EverestConfig:
        """This function exists as a temporary mechanism to default configurations that
        needs to be global in the sense that they should carry over both to ropt and ERT.
        When the proper mechanism for this is implemented this code
        should die.

        """
        defaulted_config = config.copy()
        assert defaulted_config.environment is not None

        random_seed = defaulted_config.environment.random_seed
        if random_seed is None:
            random_seed = random.randint(1, 2**30)

        defaulted_config.environment.random_seed = random_seed

        logging.getLogger(EVEREST).info("Using random seed: %d", random_seed)
        logging.getLogger(EVEREST).info(
            "To deterministically reproduce this experiment, "
            "add the above random seed to your configuration file."
        )

        return defaulted_config

    @classmethod
    def create(
        cls,
        ever_config: EverestConfig,
        simulation_callback: Optional[SimulationCallback] = None,
        optimization_callback: Optional[OptimizerCallback] = None,
    ) -> EverestRunModel:
        def default_simulation_callback(
            simulation_status: SimulationStatus | None, event: str
        ) -> str | None:
            return None

        def default_optimization_callback() -> str | None:
            return None

        ert_config = everest_to_ert_config(cls._add_defaults(ever_config))
        return cls(
            config=ert_config,
            everest_config=ever_config,
            simulation_callback=simulation_callback or default_simulation_callback,
            optimization_callback=optimization_callback
            or default_optimization_callback,
        )

    def run_experiment(
        self, evaluator_server_config: EvaluatorServerConfig, restart: bool = False
    ) -> None:
        self.log_at_startup()
        simulator = Simulator(
            self.everest_config,
            self.ert_config,
            self._storage,
            callback=self._simulation_callback,
        )

        # Initialize the ropt optimizer:
        optimizer = self._create_optimizer(simulator)

        # The SqliteStorage object is used to store optimization results from
        # Seba in an sqlite database. It reacts directly to events emitted by
        # Seba and is not called by Everest directly. The stored results are
        # accessed by Everest via separate SebaSnapshot objects.
        # This mechanism is outdated and not supported by the ropt package. It
        # is retained for now via the seba_sqlite package.
        seba_storage = SqliteStorage(  # type: ignore
            optimizer, self.everest_config.optimization_output_dir
        )

        # Run the optimization:
        optimizer_exit_code = optimizer.run().exit_code

        # Extract the best result from the storage.
        self._result = seba_storage.get_optimal_result()  # type: ignore

        if self._monitor_thread is not None:
            self._monitor_thread.stop()
            self._monitor_thread.join()
            self._monitor_thread = None

        self._exit_code = (
            "max_batch_num_reached"
            if self._max_batch_num_reached
            else optimizer_exit_code
        )

    def _handle_errors(
        self,
        batch: int,
        simulation: Any,
        realization: str,
        fm_name: str,
        error_path: str,
    ) -> None:
        fm_id = "b_{}_r_{}_s_{}_{}".format(batch, realization, simulation, fm_name)
        logger = logging.getLogger("forward_models")
        with open(error_path, "r", encoding="utf-8") as errors:
            error_str = errors.read()

        error_hash = hash(error_str)
        err_msg = "Batch: {} Realization: {} Simulation: {} Job: {} Failed {}".format(
            batch, realization, simulation, fm_name, "Error: {}\n {}"
        )

        if error_hash not in self._fm_errors:
            error_id = len(self._fm_errors)
            logger.error(err_msg.format(error_id, error_str))
            self._fm_errors.update({error_hash: {"error_id": error_id, "ids": [fm_id]}})
        elif fm_id not in self._fm_errors[error_hash]["ids"]:
            self._fm_errors[error_hash]["ids"].append(fm_id)
            error_id = self._fm_errors[error_hash]["error_id"]
            logger.error(err_msg.format(error_id, ""))

    def _simulation_callback(self, ctx: BatchContext | None) -> None:
        logging.getLogger(EVEREST).debug("Simulation callback called")
        if ctx is None:
            return
        if self._monitor_thread is not None:
            self._monitor_thread.stop()

        self._monitor_thread = _MonitorThread(
            context=ctx,
            error_callback=self._handle_errors,
            callback=self._sim_callback,
            delete_run_path=self._simulation_delete_run_path,
            display_all_jobs=self._display_all_jobs,
        )
        self._monitor_thread.start()

    def _on_before_forward_model_evaluation(
        self, _: Event, optimizer: BasicOptimizer, simulator: Simulator
    ) -> None:
        logging.getLogger(EVEREST).debug("Optimization callback called")

        if (
            self.everest_config.optimization is not None
            and self.everest_config.optimization.max_batch_num is not None
            and (
                simulator.number_of_evaluated_batches
                >= self.everest_config.optimization.max_batch_num
            )
        ):
            self._max_batch_num_reached = True
            logging.getLogger(EVEREST).info("Maximum number of batches reached")
            optimizer.abort_optimization()
        if (
            self._opt_callback is not None
            and self._opt_callback() == "stop_optimization"
        ):
            logging.getLogger(EVEREST).info("User abort requested.")
            optimizer.abort_optimization()

    def _create_optimizer(self, simulator: Simulator) -> BasicOptimizer:
        assert (
            self.everest_config.environment is not None
            and self.everest_config.environment is not None
        )

        ropt_output_folder = Path(self.everest_config.optimization_output_dir)
        ropt_evaluator_fn = simulator.create_forward_model_evaluator_function()

        # Initialize the optimizer with output tables. `min_header_len` is set
        # to ensure that all tables have the same number of header lines,
        # simplifying code that reads them as fixed width tables. `maximize` is
        # set because ropt reports minimization results, while everest wants
        # maximization results, necessitating a conversion step.
        optimizer = (
            BasicOptimizer(enopt_config=self.ropt_config, evaluator=ropt_evaluator_fn)
            .add_table(
                columns=RESULT_COLUMNS,
                path=ropt_output_folder / "results.txt",
                min_header_len=MIN_HEADER_LEN,
                maximize=True,
            )
            .add_table(
                columns=GRADIENT_COLUMNS,
                path=ropt_output_folder / "gradients.txt",
                table_type="gradients",
                min_header_len=MIN_HEADER_LEN,
                maximize=True,
            )
            .add_table(
                columns=SIMULATION_COLUMNS,
                path=ropt_output_folder / "simulations.txt",
                min_header_len=MIN_HEADER_LEN,
                maximize=True,
            )
            .add_table(
                columns=PERTURBATIONS_COLUMNS,
                path=ropt_output_folder / "perturbations.txt",
                table_type="gradients",
                min_header_len=MIN_HEADER_LEN,
                maximize=True,
            )
        )

        # Before each batch evaluation we check if we should abort:
        optimizer.add_observer(
            EventType.START_EVALUATION,
            functools.partial(
                self._on_before_forward_model_evaluation,
                optimizer=optimizer,
                simulator=simulator,
            ),
        )

        return optimizer

    @classmethod
    def name(cls) -> str:
        return "Batch simulator"

    @classmethod
    def description(cls) -> str:
        return "Run batches "

    @property
    def exit_code(
        self,
    ) -> Optional[Literal["max_batch_num_reached"] | OptimizerExitCode]:
        return self._exit_code

    @property
    def result(self) -> Optional[seba_sqlite.sqlite_storage.OptimalResult]:
        return self._result

    def __repr__(self) -> str:
        config_json = json.dumps(self.everest_config, sort_keys=True, indent=2)
        return f"EverestRunModel(config={config_json})"
