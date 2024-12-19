from __future__ import annotations

import datetime
import functools
import json
import logging
import os
import queue
import random
import shutil
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
)

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
from everest.simulator import SimulatorCache
from everest.simulator.everest_to_ert import everest_to_ert_config
from everest.strings import EVEREST

from ..run_arg import RunArg, create_run_arguments
from .base_run_model import BaseRunModel, StatusEvents

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.storage import Ensemble, Experiment


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
        self._fm_errors: dict[int, dict[str, Any]] = {}
        self._simulation_delete_run_path = (
            False
            if everest_config.simulator is None
            else (everest_config.simulator.delete_run_path or False)
        )
        self._display_all_jobs = display_all_jobs
        self._result: OptimalResult | None = None
        self._exit_code: Literal["max_batch_num_reached"] | OptimizerExitCode | None = (
            None
        )
        self._max_batch_num_reached = False
        self._simulator_cache: SimulatorCache | None = None
        if (
            everest_config.simulator is not None
            and everest_config.simulator.enable_cache
        ):
            self._simulator_cache = SimulatorCache()
        self._experiment: Experiment | None = None
        self.eval_server_cfg: EvaluatorServerConfig | None = None
        storage = open_storage(config.ens_path, mode="w")
        status_queue: queue.SimpleQueue[StatusEvents] = queue.SimpleQueue()
        self.batch_id: int = 0
        self.status: SimulationStatus | None = None

        super().__init__(
            config,
            storage,
            config.queue_config,
            status_queue,
            active_realizations=[],  # Set dynamically in run_forward_model()
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
        simulation_callback: SimulationCallback | None = None,
        optimization_callback: OptimizerCallback | None = None,
    ) -> EverestRunModel:
        def default_simulation_callback(
            simulation_status: SimulationStatus | None,
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
        self.restart = restart
        self.eval_server_cfg = evaluator_server_config
        self._experiment = self._storage.create_experiment(
            name=f"EnOpt@{datetime.datetime.now().strftime('%Y-%m-%d@%H:%M:%S')}",
            parameters=self.ert_config.ensemble_config.parameter_configuration,
            responses=self.ert_config.ensemble_config.response_configuration,
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
            optimizer, self.everest_config.optimization_output_dir
        )

        # Run the optimization:
        optimizer_exit_code = optimizer.run().exit_code

        # Extract the best result from the storage.
        self._result = OptimalResult.from_seba_optimal_result(
            seba_storage.get_optimal_result()  # type: ignore
        )

        self._exit_code = (
            "max_batch_num_reached"
            if self._max_batch_num_reached
            else optimizer_exit_code
        )

    def check_if_runpath_exists(self) -> bool:
        return (
            self.everest_config.simulation_dir is not None
            and os.path.exists(self.everest_config.simulation_dir)
            and any(os.listdir(self.everest_config.simulation_dir))
        )

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

    def _delete_runpath(self, run_args: list[RunArg]) -> None:
        logging.getLogger(EVEREST).debug("Simulation callback called")
        if self._simulation_delete_run_path:
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

    def _on_before_forward_model_evaluation(
        self, _: OptimizerEvent, optimizer: BasicOptimizer
    ) -> None:
        logging.getLogger(EVEREST).debug("Optimization callback called")

        if (
            self.everest_config.optimization is not None
            and self.everest_config.optimization.max_batch_num is not None
            and (self.batch_id >= self.everest_config.optimization.max_batch_num)
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

    def _create_optimizer(self) -> BasicOptimizer:
        assert (
            self.everest_config.environment is not None
            and self.everest_config.environment is not None
        )

        ropt_output_folder = Path(self.everest_config.optimization_output_dir)

        # Initialize the optimizer with output tables. `min_header_len` is set
        # to ensure that all tables have the same number of header lines,
        # simplifying code that reads them as fixed width tables. `maximize` is
        # set because ropt reports minimization results, while everest wants
        # maximization results, necessitating a conversion step.
        optimizer = (
            BasicOptimizer(
                enopt_config=self.ropt_config, evaluator=self._forward_model_evaluator
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

    @classmethod
    def name(cls) -> str:
        return "Optimization run"

    @classmethod
    def description(cls) -> str:
        return "Run batches "

    @property
    def exit_code(
        self,
    ) -> Literal["max_batch_num_reached"] | OptimizerExitCode | None:
        return self._exit_code

    @property
    def result(self) -> OptimalResult | None:
        return self._result

    def __repr__(self) -> str:
        config_json = json.dumps(self.everest_config, sort_keys=True, indent=2)
        return f"EverestRunModel(config={config_json})"

    @staticmethod
    def _add_control(
        controls: Mapping[str, Any],
        control_name: tuple[Any, ...],
        control_value: float,
    ) -> None:
        group_name = control_name[0]
        variable_name = control_name[1]
        group = controls[group_name]
        if len(control_name) > 2:
            index_name = str(control_name[2])
            if variable_name in group:
                group[variable_name][index_name] = control_value
            else:
                group[variable_name] = {index_name: control_value}
        else:
            group[variable_name] = control_value

    @staticmethod
    def _get_active_results(
        results: list[dict[str, NDArray[np.float64]]],
        names: tuple[str],
        controls: NDArray[np.float64],
        active: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        values = np.zeros((controls.shape[0], len(names)), dtype=float64)
        for func_idx, name in enumerate(names):
            values[active, func_idx] = np.fromiter(
                (np.nan if not result else result[name][0] for result in results),
                dtype=np.float64,
            )
        return values

    def init_case_data(
        self,
        control_values: NDArray[np.float64],
        metadata: EvaluatorContext,
        realization_ids: list[int],
    ) -> tuple[
        list[tuple[int, defaultdict[str, Any]]], NDArray[np.bool_], dict[int, int]
    ]:
        active = (
            np.ones(control_values.shape[0], dtype=np.bool_)
            if metadata.active is None
            else np.fromiter(
                (metadata.active[realization] for realization in metadata.realizations),
                dtype=np.bool_,
            )
        )
        case_data = []
        cached = {}

        for sim_idx, real_id in enumerate(realization_ids):
            if self._simulator_cache is not None:
                cache_id = self._simulator_cache.find_key(
                    real_id, control_values[sim_idx, :]
                )
                if cache_id is not None:
                    cached[sim_idx] = cache_id
                    active[sim_idx] = False

            if active[sim_idx]:
                controls: defaultdict[str, Any] = defaultdict(dict)
                assert metadata.config.variables.names is not None
                for control_name, control_value in zip(
                    metadata.config.variables.names,
                    control_values[sim_idx, :],
                    strict=False,
                ):
                    self._add_control(controls, control_name, control_value)
                case_data.append((real_id, controls))
        return case_data, active, cached

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

        if set(controls.keys()) != set(self.everest_config.control_names):
            err_msg = "Mismatch between initialized and provided control names."
            raise KeyError(err_msg)

        for control_name, control in controls.items():
            ext_config = self.ert_config.ensemble_config.parameter_configs[control_name]
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

    def _forward_model_evaluator(
        self, control_values: NDArray[np.float64], metadata: EvaluatorContext
    ) -> EvaluatorResult:
        def _slug(entity: str) -> str:
            entity = " ".join(str(entity).split())
            return "".join([x if x.isalnum() else "_" for x in entity.strip()])

        self.status = None  # Reset the current run status
        assert metadata.config.realizations.names
        realization_ids = [
            metadata.config.realizations.names[realization]
            for realization in metadata.realizations
        ]
        case_data, active, cached = self.init_case_data(
            control_values=control_values,
            metadata=metadata,
            realization_ids=realization_ids,
        )
        assert self._experiment
        ensemble = self._experiment.create_ensemble(
            name=f"batch_{self.batch_id}",
            ensemble_size=len(case_data),
        )
        for sim_id, (geo_id, controls) in enumerate(case_data):
            assert isinstance(geo_id, int)
            self._setup_sim(sim_id, controls, ensemble)

        substitutions = self.ert_config.substitutions
        # fill in the missing geo_id data
        substitutions["<BATCH_NAME>"] = _slug(ensemble.name)
        self.active_realizations = [True] * len(case_data)
        for sim_id, (geo_id, _) in enumerate(case_data):
            if self.active_realizations[sim_id]:
                substitutions[f"<GEO_ID_{sim_id}_0>"] = str(geo_id)

        run_paths = Runpaths(
            jobname_format=self.ert_config.model_config.jobname_format_string,
            runpath_format=self.ert_config.model_config.runpath_format_string,
            filename=str(self.ert_config.runpath_file),
            substitutions=substitutions,
            eclbase=self.ert_config.model_config.eclbase_format_string,
        )
        run_args = create_run_arguments(
            run_paths,
            self.active_realizations,
            ensemble=ensemble,
        )

        self._context_env.update(
            {
                "_ERT_EXPERIMENT_ID": str(ensemble.experiment_id),
                "_ERT_ENSEMBLE_ID": str(ensemble.id),
                "_ERT_SIMULATION_MODE": "batch_simulation",
            }
        )
        assert self.eval_server_cfg
        self._evaluate_and_postprocess(run_args, ensemble, self.eval_server_cfg)

        self._delete_runpath(run_args)
        # gather results
        results: list[dict[str, npt.NDArray[np.float64]]] = []
        for sim_id, successful in enumerate(self.active_realizations):
            if not successful:
                logger.error(f"Simulation {sim_id} failed.")
                results.append({})
                continue
            d = {}
            for key in self.everest_config.result_names:
                data = ensemble.load_responses(key, (sim_id,))
                d[key] = data["values"].to_numpy()
            results.append(d)

        for fnc_name, alias in self.everest_config.function_aliases.items():
            for result in results:
                result[fnc_name] = result[alias]

        objectives = self._get_active_results(
            results,
            metadata.config.objectives.names,  # type: ignore
            control_values,
            active,
        )

        constraints = None
        if metadata.config.nonlinear_constraints is not None:
            constraints = self._get_active_results(
                results,
                metadata.config.nonlinear_constraints.names,  # type: ignore
                control_values,
                active,
            )

        if self._simulator_cache is not None:
            for sim_idx, cache_id in cached.items():
                objectives[sim_idx, ...] = self._simulator_cache.get_objectives(
                    cache_id
                )
                if constraints is not None:
                    constraints[sim_idx, ...] = self._simulator_cache.get_constraints(
                        cache_id
                    )

        sim_ids = np.empty(control_values.shape[0], dtype=np.intc)
        sim_ids.fill(-1)
        sim_ids[active] = np.arange(len(results), dtype=np.intc)

        # Add the results from active simulations to the cache:
        if self._simulator_cache is not None:
            for sim_idx, real_id in enumerate(realization_ids):
                if active[sim_idx]:
                    self._simulator_cache.add_simulation_results(
                        sim_idx, real_id, control_values, objectives, constraints
                    )

        # Note the negative sign for the objective results. Everest aims to do a
        # maximization, while the standard practice of minimizing is followed by
        # ropt. Therefore we will minimize the negative of the objectives:
        evaluator_result = EvaluatorResult(
            objectives=-objectives,
            constraints=constraints,
            batch_id=self.batch_id,
            evaluation_ids=sim_ids,
        )

        self.batch_id += 1

        return evaluator_result

    def send_snapshot_event(self, event: Event, iteration: int) -> None:
        super().send_snapshot_event(event, iteration)
        if type(event) in {EESnapshot, EESnapshotUpdate}:
            newstatus = self._simulation_status(self.get_current_snapshot())
            if self.status != newstatus:  # No change in status
                self._sim_callback(newstatus)
                self.status = newstatus

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
                    batch=self.batch_id,
                    simulation=simulation,
                    realization=realization,
                    fm_name=fm_step.get("name", "Unknwon"),  # type: ignore
                    error_path=fm_step.get("stderr", ""),  # type: ignore
                    fm_running_err=fm_step.get("error", ""),  # type: ignore
                )
        jobs_progress.append(jobs)

        return {
            "status": self.get_current_status(),
            "progress": jobs_progress,
            "batch_number": self.batch_id,
        }
