from __future__ import annotations

import copy
import datetime
import json
import logging
import os
import random
import re
import shutil
import threading
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypedDict

from ropt.enums import EventType
from ropt.plan import OptimizationPlanRunner
from seba_sqlite import SqliteStorage

import everest
from everest.config import EverestConfig
from everest.optimizer.everest2ropt import everest2ropt
from everest.plugins.site_config_env import PluginSiteConfigEnv
from everest.simulator import Simulator
from everest.strings import EVEREST, SIMULATOR_END, SIMULATOR_START, SIMULATOR_UPDATE
from everest.util import makedirs_if_needed

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
    "evaluations.unscaled_variables": "Control",
    "linear_constraints.values": "IC-diff",
    "nonlinear_constraints.values": "OC-diff",
    "functions.scaled_objectives": "Scaled-Objective",
    "functions.scaled_constraints": "Scaled-Constraint",
    "evaluations.variables": "Scaled-Control",
    "nonlinear_constraints.scaled_values": "Scaled-OC-diff",
    "nonlinear_constraints.scaled_violations": "Scaled-OC-violation",
    "metadata.restart": "Restart",
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
    "evaluations.unscaled_variables": "Control",
    "evaluations.objectives": "Objective",
    "evaluations.constraints": "Constraint",
    "evaluations.variables": "Scaled-Control",
    "evaluations.scaled_objectives": "Scaled-Objective",
    "evaluations.scaled_constraints": "Scaled-Constraint",
}
PERTURBATIONS_COLUMNS = {
    "result_id": "ID",
    "batch_id": "Batch",
    "realization": "Realization",
    "evaluations.perturbed_evaluation_ids": "Simulation",
    "evaluations.unscaled_perturbed_variables": "Control",
    "evaluations.perturbed_objectives": "Objective",
    "evaluations.perturbed_constraints": "Constraint",
    "evaluations.perturbed_variables": "Scaled-Control",
    "evaluations.scaled_perturbed_objectives": "Scaled-Objective",
    "evaluations.scaled_perturbed_constraints": "Scaled-Constraint",
}
MIN_HEADER_LEN = 3


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


def start_optimization(
    config, simulation_callback=None, optimization_callback=None, display_all_jobs=True
):
    workflow = _EverestWorkflow(
        config, simulation_callback, optimization_callback, display_all_jobs
    )
    with PluginSiteConfigEnv():
        res = workflow.start_optimization()
    return res


def _add_defaults(config: EverestConfig):
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
        callback: Optional[Callable[[SimulationStatus], Any]] = None,
        error_callback=None,
        delete_run_path=False,
        display_all_jobs=False,
    ):
        super(_MonitorThread, self).__init__()

        # temporarily living simulation context
        self._context = context
        self._callback = callback if callback is not None else lambda *_, **__: None
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

                        def onerror(_, path, sys_info):
                            logging.getLogger(EVEREST).debug(
                                "Failed to remove {}, {}".format(path, sys_info)
                            )

                        shutil.rmtree(path_to_delete, onerror=onerror)  # pylint: disable=deprecated-argument

        self._context = None
        self._callback = lambda *_, **__: None
        self._shutdown_flag = True

    @property
    def _batch_number(self):
        """
        Return the current batch number from context.

        """
        # Get the string name of current case
        batch_n_sim_string = self._context.get_ensemble().name

        search = re.search(r"batch_([0-9]+)", batch_n_sim_string)
        return search.groups()[-1] if search is not None else "N/A"

    def _simulation_status(self) -> SimulationStatus:
        assert self._context is not None

        def extract(path_str, key):
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
                        and fms.name in everest.jobs.shell_commands
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
                        self._error_callback(
                            batch_number,
                            simulation,
                            realization,
                            fms.name,
                            fms.std_err_file,
                        )
                jobs_progress.append(jobs)
        return {
            "status": copy.deepcopy(self._context.status),
            "progress": jobs_progress,
            "batch_number": batch_number,
        }

    def run(self):
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

    def stop(self):
        if self._context is not None:
            self._context.stop()
        self._shutdown_flag = True


class _EverestWorkflow(object):
    """
    An instance of this class is the main object in everest.

    Through this object an optimization experiment is instantiated and executed/run.
    This object will provide access to the entire optimization configuration.
    """

    def __init__(
        self,
        config: EverestConfig,
        simulation_callback=None,
        optimization_callback=None,
        display_all_jobs=True,
    ):
        """Will initialize an Everest instance either from a configuration file or
        a loaded config.

        @config   a dictionary containing the configuration.  See everest --doc
                  for documentation on the config

        @callback a function that will be called whenever changes in the
                  simulation or optimization routine occur, e.g., when one
                  realization's simulation completes, the status vector will be
                  sent, with the event SIMULATOR_UPDATE.
        """

        # Callbacks
        self._sim_callback = simulation_callback
        self._opt_callback = optimization_callback

        self._monitor_thread = None  # Thread for monitoring simulator activity

        self._config = _add_defaults(config)

        makedirs_if_needed(self.config.log_dir)
        makedirs_if_needed(self.config.optimization_output_dir)

        self._simulation_delete_run_path = (
            False
            if config.simulator is None
            else (config.simulator.delete_run_path or False)
        )

        self._display_all_jobs = display_all_jobs
        self._fm_errors: Dict[str, Dict[str, Any]] = {}
        self._max_batch_num_reached = False

    def _handle_errors(self, batch, simulation, realization, fm_name, error_path):
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

    def _simulation_callback(self, *args, **_):
        logging.getLogger(EVEREST).debug("Simulation callback called")
        ctx = args[0]
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

    def _ropt_callback(self, event, optimizer, simulator):
        logging.getLogger(EVEREST).debug("Optimization callback called")

        if self._config.optimization.max_batch_num is not None and (
            simulator.number_of_evaluated_batches
            >= self._config.optimization.max_batch_num
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

    def start_optimization(self):
        """Run an optimization with the current settings.

        This method must be called from the same thread where this
        object has been created (probably because of the use of sqlite3
        deeper down).
        This method is not thread safe. Multiple overlapping executions
        of this method will probably lead to a crash
        """
        assert self._monitor_thread is None

        # Initialize the Everest simulator:
        simulator = Simulator(self.config, callback=self._simulation_callback)

        # Initialize the ropt optimizer:
        optimizer = self._configure_optimizer(simulator)

        # Before each batch evaluation we check if we should abort:
        optimizer.add_observer(
            EventType.START_EVALUATION,
            partial(self._ropt_callback, optimizer=optimizer, simulator=simulator),
        )

        # The SqliteStorage object is used to store optimization results from
        # Seba in an sqlite database. It reacts directly to events emitted by
        # Seba and is not called by Everest directly. The stored results are
        # accessed by Everest via separate SebaSnapshot objects.
        # This mechanism is outdated and not supported by the ropt package. It
        # is retained for now via the seba_sqlite package.
        seba_storage = SqliteStorage(optimizer, self.config.optimization_output_dir)

        # Run the optimization:
        exit_code = optimizer.run().exit_code

        # Extract the best result from the storage.
        self._result = seba_storage.get_optimal_result()

        if self._monitor_thread is not None:
            self._monitor_thread.stop()
            self._monitor_thread.join()
            self._monitor_thread = None

        return "max_batch_num_reached" if self._max_batch_num_reached else exit_code

    @property
    def result(self):
        return self._result

    @property
    def config(self) -> EverestConfig:
        return self._config

    def __repr__(self):
        return "EverestWorkflow(config=%s)" % json.dumps(
            self.config, sort_keys=True, indent=2
        )

    def _configure_optimizer(self, simulator: Simulator) -> OptimizationPlanRunner:
        optimizer = OptimizationPlanRunner(
            enopt_config=everest2ropt(self.config),
            evaluator=simulator,
            seed=self._config.environment.random_seed,
        )

        # Initialize output tables. `min_header_len` is set to ensure that all
        # tables have the same number of header lines, simplifying code that
        # reads them as fixed width tables. `maximize` is set because ropt
        # reports minimization results, while everest wants maximization
        # results, necessitating a conversion step.
        ropt_output_folder = Path(self.config.optimization_output_dir)
        optimizer.add_table(
            columns=RESULT_COLUMNS,
            path=ropt_output_folder / "results.txt",
            min_header_len=MIN_HEADER_LEN,
            maximize=True,
        )
        optimizer.add_table(
            columns=GRADIENT_COLUMNS,
            path=ropt_output_folder / "gradients.txt",
            table_type="gradients",
            min_header_len=MIN_HEADER_LEN,
            maximize=True,
        )
        optimizer.add_table(
            columns=SIMULATION_COLUMNS,
            path=ropt_output_folder / "simulations.txt",
            min_header_len=MIN_HEADER_LEN,
            maximize=True,
        )
        optimizer.add_table(
            columns=PERTURBATIONS_COLUMNS,
            path=ropt_output_folder / "perturbations.txt",
            table_type="gradients",
            min_header_len=MIN_HEADER_LEN,
            maximize=True,
        )
        return optimizer
