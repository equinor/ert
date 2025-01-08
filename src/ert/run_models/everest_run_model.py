from __future__ import annotations

import datetime
import functools
import json
import logging
import os
import queue
import shutil
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import seba_sqlite.sqlite_storage
from numpy import float64
from numpy._typing import NDArray
from ropt.enums import EventType, OptimizerExitCode
from ropt.evaluator import EvaluatorContext, EvaluatorResult
from ropt.plan import BasicOptimizer
from ropt.plan import Event as OptimizerEvent
from seba_sqlite import SqliteStorage
from typing_extensions import TypedDict

from _ert.events import EESnapshot, EESnapshotUpdate, Event
from ert.config import ErtConfig, ExtParamConfig
from ert.ensemble_evaluator import EnsembleSnapshot, EvaluatorServerConfig
from ert.runpaths import Runpaths
from ert.storage import open_storage
from everest.config import EverestConfig
from everest.optimizer.everest2ropt import everest2ropt
from everest.simulator.everest_to_ert import everest_to_ert_config
from everest.strings import EVEREST

from ..run_arg import RunArg, create_run_arguments
from .base_run_model import BaseRunModel, StatusEvents

if TYPE_CHECKING:
    from ert.storage import Ensemble, Experiment


logger = logging.getLogger(__name__)


class SimulationStatus(TypedDict):
    status: dict[str, int]
    progress: list[list[JobProgress]]
    batch_number: int


class JobProgress(TypedDict):
    name: str
    status: str
    error: str | None
    start_time: datetime.datetime | None
    end_time: datetime.datetime | None
    realization: str
    simulation: str


class SimulationCallback(Protocol):
    def __call__(self, simulation_status: SimulationStatus | None) -> str | None: ...


class OptimizerCallback(Protocol):
    def __call__(self) -> str | None: ...


@dataclass
class OptimalResult:
    batch: int
    controls: list[Any]
    total_objective: float

    @staticmethod
    def from_seba_optimal_result(
        o: seba_sqlite.sqlite_storage.OptimalResult | None = None,
    ) -> OptimalResult | None:
        if o is None:
            return None

        return OptimalResult(
            batch=o.batch, controls=o.controls, total_objective=o.total_objective
        )


class EverestExitCode(IntEnum):
    COMPLETED = 1
    TOO_FEW_REALIZATIONS = 2
    MAX_FUNCTIONS_REACHED = 3
    MAX_BATCH_NUM_REACHED = 4
    USER_ABORT = 5


class EverestRunModel(BaseRunModel):
    def __init__(
        self,
        config: ErtConfig,
        everest_config: EverestConfig,
        simulation_callback: SimulationCallback | None,
        optimization_callback: OptimizerCallback | None,
    ):
        Path(everest_config.log_dir).mkdir(parents=True, exist_ok=True)
        Path(everest_config.optimization_output_dir).mkdir(parents=True, exist_ok=True)

        assert everest_config.environment is not None
        logging.getLogger(EVEREST).info(
            "Using random seed: %d. To deterministically reproduce this experiment, "
            "add the above random seed to your configuration file.",
            everest_config.environment.random_seed,
        )

        self._everest_config = everest_config
        self._ropt_config = everest2ropt(everest_config)

        self._sim_callback = simulation_callback
        self._opt_callback = optimization_callback
        self._fm_errors: dict[int, dict[str, Any]] = {}
        self._result: OptimalResult | None = None
        self._exit_code: EverestExitCode | None = None
        self._simulator_cache = (
            SimulatorCache()
            if (
                everest_config.simulator is not None
                and everest_config.simulator.enable_cache
            )
            else None
        )
        self._experiment: Experiment | None = None
        self._eval_server_cfg: EvaluatorServerConfig | None = None
        self._batch_id: int = 0
        self._status: SimulationStatus | None = None

        storage = open_storage(config.ens_path, mode="w")
        status_queue: queue.SimpleQueue[StatusEvents] = queue.SimpleQueue()

        super().__init__(
            storage,
            config.runpath_file,
            Path(config.user_config_file),
            config.env_vars,
            config.env_pr_fm_step,
            config.model_config,
            config.queue_config,
            config.forward_model_steps,
            status_queue,
            config.substitutions,
            config.ert_templates,
            config.hooked_workflows,
            active_realizations=[],  # Set dynamically in run_forward_model()
        )
        self.support_restart = False
        self._parameter_configuration = config.ensemble_config.parameter_configuration
        self._parameter_configs = config.ensemble_config.parameter_configs
        self._response_configuration = config.ensemble_config.response_configuration

    @classmethod
    def create(
        cls,
        ever_config: EverestConfig,
        simulation_callback: SimulationCallback | None = None,
        optimization_callback: OptimizerCallback | None = None,
    ) -> EverestRunModel:
        return cls(
            config=everest_to_ert_config(ever_config),
            everest_config=ever_config,
            simulation_callback=simulation_callback,
            optimization_callback=optimization_callback,
        )

    @classmethod
    def name(cls) -> str:
        return "Optimization run"

    @classmethod
    def description(cls) -> str:
        return "Run batches "

    @property
    def exit_code(self) -> EverestExitCode | None:
        return self._exit_code

    @property
    def result(self) -> OptimalResult | None:
        return self._result

    def __repr__(self) -> str:
        config_json = json.dumps(self._everest_config, sort_keys=True, indent=2)
        return f"EverestRunModel(config={config_json})"

    def run_experiment(
        self, evaluator_server_config: EvaluatorServerConfig, restart: bool = False
    ) -> None:
        self.log_at_startup()
        self._eval_server_cfg = evaluator_server_config
        self._experiment = self._storage.create_experiment(
            name=f"EnOpt@{datetime.datetime.now().strftime('%Y-%m-%d@%H:%M:%S')}",
            parameters=self._parameter_configuration,
            responses=self._response_configuration,
        )

        # Initialize the ropt optimizer:
        optimizer = self._create_optimizer()

        # The SqliteStorage object is used to store optimization results from
        # Seba in an sqlite database. It reacts directly to events emitted by
        # Seba and is not called by Everest directly. The stored results are
        # accessed by Everest via separate SebaSnapshot objects.
        # This mechanism is outdated and not supported by the ropt package. It
        # is retained for now via the seba_sqlite package.
        seba_storage = SqliteStorage(  # type: ignore
            optimizer, self._everest_config.optimization_output_dir
        )

        # Run the optimization:
        optimizer_exit_code = optimizer.run().exit_code

        # Extract the best result from the storage.
        self._result = OptimalResult.from_seba_optimal_result(
            seba_storage.get_optimal_result()  # type: ignore
        )

        if self._exit_code is None:
            match optimizer_exit_code:
                case OptimizerExitCode.MAX_FUNCTIONS_REACHED:
                    self._exit_code = EverestExitCode.MAX_FUNCTIONS_REACHED
                case OptimizerExitCode.USER_ABORT:
                    self._exit_code = EverestExitCode.USER_ABORT
                case OptimizerExitCode.TOO_FEW_REALIZATIONS:
                    self._exit_code = EverestExitCode.TOO_FEW_REALIZATIONS
                case _:
                    self._exit_code = EverestExitCode.COMPLETED

    def _create_optimizer(self) -> BasicOptimizer:
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

        # Initialize the optimizer with output tables. `min_header_len` is set
        # to ensure that all tables have the same number of header lines,
        # simplifying code that reads them as fixed width tables. `maximize` is
        # set because ropt reports minimization results, while everest wants
        # maximization results, necessitating a conversion step.
        ropt_output_folder = Path(self._everest_config.optimization_output_dir)
        optimizer = (
            BasicOptimizer(
                enopt_config=self._ropt_config, evaluator=self._forward_model_evaluator
            )
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
            ),
        )

        return optimizer

    def _on_before_forward_model_evaluation(
        self, _: OptimizerEvent, optimizer: BasicOptimizer
    ) -> None:
        logging.getLogger(EVEREST).debug("Optimization callback called")

        if (
            self._everest_config.optimization is not None
            and self._everest_config.optimization.max_batch_num is not None
            and (self._batch_id >= self._everest_config.optimization.max_batch_num)
        ):
            self._exit_code = EverestExitCode.MAX_BATCH_NUM_REACHED
            logging.getLogger(EVEREST).info("Maximum number of batches reached")
            optimizer.abort_optimization()
        if (
            self._opt_callback is not None
            and self._opt_callback() == "stop_optimization"
        ):
            logging.getLogger(EVEREST).info("User abort requested.")
            optimizer.abort_optimization()

    def _forward_model_evaluator(
        self, control_values: NDArray[np.float64], evaluator_context: EvaluatorContext
    ) -> EvaluatorResult:
        # Reset the current run status:
        self._status = None

        # Get cached_results:
        cached_results = self._get_cached_results(control_values, evaluator_context)

        # Create the batch to run:
        batch_data = self._init_batch_data(
            control_values, evaluator_context, cached_results
        )

        # Initialize a new ensemble in storage:
        assert self._experiment is not None
        ensemble = self._experiment.create_ensemble(
            name=f"batch_{self._batch_id}",
            ensemble_size=len(batch_data),
        )
        for sim_id, controls in enumerate(batch_data.values()):
            self._setup_sim(sim_id, controls, ensemble)

        # Evaluate the batch:
        run_args = self._get_run_args(ensemble, evaluator_context, batch_data)
        self._context_env.update(
            {
                "_ERT_EXPERIMENT_ID": str(ensemble.experiment_id),
                "_ERT_ENSEMBLE_ID": str(ensemble.id),
                "_ERT_SIMULATION_MODE": "batch_simulation",
            }
        )
        assert self._eval_server_cfg is not None
        self._evaluate_and_postprocess(run_args, ensemble, self._eval_server_cfg)

        # If necessary, delete the run path:
        self._delete_runpath(run_args)

        # Gather the results and create the result for ropt:
        results = self._gather_simulation_results(ensemble)
        evaluator_result = self._make_evaluator_result(
            control_values, batch_data, results, cached_results
        )

        # Add the results from the evaluations to the cache:
        self._add_results_to_cache(
            control_values,
            evaluator_context,
            batch_data,
            evaluator_result.objectives,
            evaluator_result.constraints,
        )

        # Increase the batch ID for the next evaluation:
        self._batch_id += 1

        return evaluator_result

    def _get_cached_results(
        self, control_values: NDArray[np.float64], evaluator_context: EvaluatorContext
    ) -> dict[int, Any]:
        cached_results: dict[int, Any] = {}
        if self._simulator_cache is not None:
            for control_idx, real_idx in enumerate(evaluator_context.realizations):
                cached_data = self._simulator_cache.get(
                    self._everest_config.model.realizations[real_idx],
                    control_values[control_idx, :],
                )
                if cached_data is not None:
                    cached_results[control_idx] = cached_data
        return cached_results

    def _init_batch_data(
        self,
        control_values: NDArray[np.float64],
        evaluator_context: EvaluatorContext,
        cached_results: dict[int, Any],
    ) -> dict[int, dict[str, Any]]:
        def add_control(
            controls: dict[str, Any],
            control_name: tuple[Any, ...],
            control_value: float,
        ) -> None:
            group_name = control_name[0]
            variable_name = control_name[1]
            group = controls.get(group_name, {})
            if len(control_name) > 2:
                index_name = str(control_name[2])
                if variable_name in group:
                    group[variable_name][index_name] = control_value
                else:
                    group[variable_name] = {index_name: control_value}
            else:
                group[variable_name] = control_value
            controls[group_name] = group

        batch_data = {}
        for control_idx in range(control_values.shape[0]):
            if control_idx not in cached_results and (
                evaluator_context.active is None
                or evaluator_context.active[evaluator_context.realizations[control_idx]]
            ):
                controls: dict[str, Any] = {}
                for control_name, control_value in zip(
                    self._everest_config.control_name_tuples,
                    control_values[control_idx, :],
                    strict=False,
                ):
                    add_control(controls, control_name, control_value)
                batch_data[control_idx] = controls
        return batch_data

    def _setup_sim(
        self,
        sim_id: int,
        controls: dict[str, dict[str, Any]],
        ensemble: Ensemble,
    ) -> None:
        def _check_suffix(
            ext_config: ExtParamConfig,
            key: str,
            assignment: dict[str, Any] | tuple[str, str] | str | int,
        ) -> None:
            if key not in ext_config:
                raise KeyError(f"No such key: {key}")
            if isinstance(assignment, dict):  # handle suffixes
                suffixes = ext_config[key]
                if len(assignment) != len(suffixes):
                    missingsuffixes = set(suffixes).difference(set(assignment.keys()))
                    raise KeyError(
                        f"Key {key} is missing values for "
                        f"these suffixes: {missingsuffixes}"
                    )
                for suffix in assignment:
                    if suffix not in suffixes:
                        raise KeyError(
                            f"Key {key} has suffixes {suffixes}. "
                            f"Can't find the requested suffix {suffix}"
                        )
            else:
                suffixes = ext_config[key]
                if suffixes:
                    raise KeyError(
                        f"Key {key} has suffixes, a suffix must be specified"
                    )

        if set(controls.keys()) != set(self._everest_config.control_names):
            err_msg = "Mismatch between initialized and provided control names."
            raise KeyError(err_msg)

        for control_name, control in controls.items():
            ext_config = self._parameter_configs[control_name]
            if isinstance(ext_config, ExtParamConfig):
                if len(ext_config) != len(control.keys()):
                    raise KeyError(
                        f"Expected {len(ext_config)} variables for "
                        f"control {control_name}, "
                        f"received {len(control.keys())}."
                    )
                for var_name, var_setting in control.items():
                    _check_suffix(ext_config, var_name, var_setting)

                ensemble.save_parameters(
                    control_name, sim_id, ExtParamConfig.to_dataset(control)
                )

    def _get_run_args(
        self,
        ensemble: Ensemble,
        evaluator_context: EvaluatorContext,
        batch_data: dict[int, Any],
    ) -> list[RunArg]:
        substitutions = self._substitutions
        substitutions["<BATCH_NAME>"] = ensemble.name
        self.active_realizations = [True] * len(batch_data)
        for sim_id, control_idx in enumerate(batch_data.keys()):
            substitutions[f"<GEO_ID_{sim_id}_0>"] = str(
                self._everest_config.model.realizations[
                    evaluator_context.realizations[control_idx]
                ]
            )
        run_paths = Runpaths(
            jobname_format=self._model_config.jobname_format_string,
            runpath_format=self._model_config.runpath_format_string,
            filename=str(self._runpath_file),
            substitutions=substitutions,
            eclbase=self._model_config.eclbase_format_string,
        )
        return create_run_arguments(
            run_paths,
            self.active_realizations,
            ensemble=ensemble,
        )

    def _delete_runpath(self, run_args: list[RunArg]) -> None:
        logging.getLogger(EVEREST).debug("Simulation callback called")
        if (
            self._everest_config.simulator is not None
            and self._everest_config.simulator.delete_run_path
        ):
            for i, real in self.get_current_snapshot().reals.items():
                path_to_delete = run_args[int(i)].runpath
                if real["status"] == "Finished" and os.path.isdir(path_to_delete):

                    def onerror(
                        _: Callable[..., Any],
                        path: str,
                        sys_info: tuple[
                            type[BaseException], BaseException, TracebackType
                        ],
                    ) -> None:
                        logging.getLogger(EVEREST).debug(
                            f"Failed to remove {path}, {sys_info}"
                        )

                    shutil.rmtree(path_to_delete, onerror=onerror)  # pylint: disable=deprecated-argument

    def _gather_simulation_results(
        self, ensemble: Ensemble
    ) -> list[dict[str, NDArray[np.float64]]]:
        results: list[dict[str, NDArray[np.float64]]] = []
        for sim_id, successful in enumerate(self.active_realizations):
            if not successful:
                logger.error(f"Simulation {sim_id} failed.")
                results.append({})
                continue
            d = {}
            for key in self._everest_config.result_names:
                data = ensemble.load_responses(key, (sim_id,))
                d[key] = data["values"].to_numpy()
            results.append(d)
        for fnc_name, alias in self._everest_config.function_aliases.items():
            for result in results:
                result[fnc_name] = result[alias]
        return results

    def _make_evaluator_result(
        self,
        control_values: NDArray[np.float64],
        batch_data: dict[int, Any],
        results: list[dict[str, NDArray[np.float64]]],
        cached_results: dict[int, Any],
    ) -> EvaluatorResult:
        # We minimize the negative of the objectives:
        objectives = -self._get_simulation_results(
            results, self._everest_config.objective_names, control_values, batch_data
        )

        constraints = None
        if self._everest_config.output_constraints:
            constraints = self._get_simulation_results(
                results,
                self._everest_config.constraint_names,
                control_values,
                batch_data,
            )

        if self._simulator_cache is not None:
            for control_idx, (
                cached_objectives,
                cached_constraints,
            ) in cached_results.items():
                objectives[control_idx, ...] = cached_objectives
                if constraints is not None:
                    assert cached_constraints is not None
                    constraints[control_idx, ...] = cached_constraints

        sim_ids = np.full(control_values.shape[0], -1, dtype=np.intc)
        sim_ids[list(batch_data.keys())] = np.arange(len(batch_data), dtype=np.intc)
        return EvaluatorResult(
            objectives=objectives,
            constraints=constraints,
            batch_id=self._batch_id,
            evaluation_ids=sim_ids,
        )

    @staticmethod
    def _get_simulation_results(
        results: list[dict[str, NDArray[np.float64]]],
        names: list[str],
        controls: NDArray[np.float64],
        batch_data: dict[int, Any],
    ) -> NDArray[np.float64]:
        control_indices = list(batch_data.keys())
        values = np.zeros((controls.shape[0], len(names)), dtype=float64)
        for func_idx, name in enumerate(names):
            values[control_indices, func_idx] = np.fromiter(
                (np.nan if not result else result[name][0] for result in results),
                dtype=np.float64,
            )
        return values

    def _add_results_to_cache(
        self,
        control_values: NDArray[np.float64],
        evaluator_context: EvaluatorContext,
        batch_data: dict[int, Any],
        objectives: NDArray[np.float64],
        constraints: NDArray[np.float64] | None,
    ) -> None:
        if self._simulator_cache is not None:
            for control_idx in batch_data:
                self._simulator_cache.add(
                    self._everest_config.model.realizations[
                        evaluator_context.realizations[control_idx]
                    ],
                    control_values[control_idx, ...],
                    objectives[control_idx, ...],
                    None if constraints is None else constraints[control_idx, ...],
                )

    def check_if_runpath_exists(self) -> bool:
        return (
            self._everest_config.simulation_dir is not None
            and os.path.exists(self._everest_config.simulation_dir)
            and any(os.listdir(self._everest_config.simulation_dir))
        )

    def send_snapshot_event(self, event: Event, iteration: int) -> None:
        super().send_snapshot_event(event, iteration)
        if type(event) in {EESnapshot, EESnapshotUpdate}:
            newstatus = self._simulation_status(self.get_current_snapshot())
            if self._status != newstatus:  # No change in status
                if self._sim_callback is not None:
                    self._sim_callback(newstatus)
                self._status = newstatus

    def _simulation_status(self, snapshot: EnsembleSnapshot) -> SimulationStatus:
        jobs_progress: list[list[JobProgress]] = []
        prev_realization = None
        jobs: list[JobProgress] = []
        for (realization, simulation), fm_step in snapshot.get_all_fm_steps().items():
            if realization != prev_realization:
                prev_realization = realization
                if jobs:
                    jobs_progress.append(jobs)
                jobs = []
            jobs.append(
                {
                    "name": fm_step.get("name") or "Unknown",
                    "status": fm_step.get("status") or "Unknown",
                    "error": fm_step.get("error", ""),
                    "start_time": fm_step.get("start_time", None),
                    "end_time": fm_step.get("end_time", None),
                    "realization": realization,
                    "simulation": simulation,
                }
            )
            if fm_step.get("error", ""):
                self._handle_errors(
                    batch=self._batch_id,
                    simulation=simulation,
                    realization=realization,
                    fm_name=fm_step.get("name", "Unknown"),  # type: ignore
                    error_path=fm_step.get("stderr", ""),  # type: ignore
                    fm_running_err=fm_step.get("error", ""),  # type: ignore
                )
        jobs_progress.append(jobs)

        return {
            "status": self.get_current_status(),
            "progress": jobs_progress,
            "batch_number": self._batch_id,
        }

    def _handle_errors(
        self,
        batch: int,
        simulation: Any,
        realization: str,
        fm_name: str,
        error_path: str,
        fm_running_err: str,
    ) -> None:
        fm_id = f"b_{batch}_r_{realization}_s_{simulation}_{fm_name}"
        fm_logger = logging.getLogger("forward_models")
        if Path(error_path).is_file():
            error_str = Path(error_path).read_text(encoding="utf-8") or fm_running_err
        else:
            error_str = fm_running_err
        error_hash = hash(error_str)
        err_msg = "Batch: {} Realization: {} Simulation: {} Job: {} Failed {}".format(
            batch, realization, simulation, fm_name, "\n Error: {} ID:{}"
        )

        if error_hash not in self._fm_errors:
            error_id = len(self._fm_errors)
            fm_logger.error(err_msg.format(error_str, error_id))
            self._fm_errors.update({error_hash: {"error_id": error_id, "ids": [fm_id]}})
        elif fm_id not in self._fm_errors[error_hash]["ids"]:
            self._fm_errors[error_hash]["ids"].append(fm_id)
            error_id = self._fm_errors[error_hash]["error_id"]
            fm_logger.error(err_msg.format("Already reported as", error_id))


class SimulatorCache:
    EPS = float(np.finfo(np.float32).eps)

    def __init__(self) -> None:
        self._data: defaultdict[
            int,
            list[
                tuple[
                    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None
                ]
            ],
        ] = defaultdict(list)

    def add(
        self,
        realization: int,
        control_values: NDArray[np.float64],
        objectives: NDArray[np.float64],
        constraints: NDArray[np.float64] | None,
    ) -> None:
        """Add objective and constraints for a given realization and control values.

        The realization is the index of the realization in the ensemble, as specified
        in by the realizations entry in the everest model configuration. Both the control
        values and the realization are used as keys to retrieve the objectives and
        constraints later.
        """
        self._data[realization].append(
            (
                control_values.copy(),
                objectives.copy(),
                None if constraints is None else constraints.copy(),
            ),
        )

    def get(
        self, realization: int, controls: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None] | None:
        """Get objective and constraints for a given realization and control values.

        The realization is the index of the realization in the ensemble, as specified
        in by the realizations entry in the everest model configuration. Both the control
        values and the realization are used as keys to retrieve the objectives and
        constraints from the cached values.
        """
        for control_values, objectives, constraints in self._data.get(realization, []):
            if np.allclose(controls, control_values, rtol=0.0, atol=self.EPS):
                return objectives, constraints
        return None
