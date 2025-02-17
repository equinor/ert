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
from enum import IntEnum
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy import float64
from numpy._typing import NDArray
from ropt.enums import EventType, OptimizerExitCode
from ropt.evaluator import EvaluatorContext, EvaluatorResult
from ropt.plan import BasicOptimizer
from ropt.plan import Event as OptimizerEvent
from ropt.results import FunctionResults
from ropt.transforms import OptModelTransforms
from typing_extensions import TypedDict

from _ert.events import EESnapshot, EESnapshotUpdate, Event
from ert.config import ExtParamConfig
from ert.config.ert_config import (
    read_templates,
    workflows_from_dict,
)
from ert.config.model_config import ModelConfig
from ert.config.queue_config import QueueConfig
from ert.ensemble_evaluator import EnsembleSnapshot, EvaluatorServerConfig
from ert.runpaths import Runpaths
from ert.storage import open_storage
from everest.config import ControlConfig, ControlVariableGuessListConfig, EverestConfig
from everest.config.utils import FlattenedControls
from everest.everest_storage import EverestStorage, OptimalResult
from everest.optimizer.everest2ropt import everest2ropt
from everest.optimizer.opt_model_transforms import (
    ConstraintScaler,
    ObjectiveScaler,
    get_optimization_domain_transforms,
)
from everest.simulator.everest_to_ert import (
    everest_to_ert_config_dict,
    get_ensemble_config,
    get_forward_model_steps,
    get_substitutions,
)
from everest.strings import EVEREST, STORAGE_DIR

from ..run_arg import RunArg, create_run_arguments
from .base_run_model import BaseRunModel, StatusEvents
from .event import (
    EverestBatchResultEvent,
    EverestStatusEvent,
)

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


class EverestExitCode(IntEnum):
    COMPLETED = 1
    TOO_FEW_REALIZATIONS = 2
    MAX_FUNCTIONS_REACHED = 3
    MAX_BATCH_NUM_REACHED = 4
    USER_ABORT = 5


class EverestRunModel(BaseRunModel):
    def __init__(
        self,
        everest_config: EverestConfig,
        simulation_callback: SimulationCallback | None,
        optimization_callback: OptimizerCallback | None,
        status_queue: queue.SimpleQueue[StatusEvents] | None = None,
    ):
        assert everest_config.log_dir is not None
        assert everest_config.optimization_output_dir is not None

        Path(everest_config.log_dir).mkdir(parents=True, exist_ok=True)
        Path(everest_config.optimization_output_dir).mkdir(parents=True, exist_ok=True)

        assert everest_config.environment is not None
        logging.getLogger(EVEREST).info(
            "Using random seed: %d. To deterministically reproduce this experiment, "
            "add the above random seed to your configuration file.",
            everest_config.environment.random_seed,
        )

        self._everest_config = everest_config
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

        ens_path = os.path.join(everest_config.output_dir, STORAGE_DIR)
        storage = open_storage(ens_path, mode="w")

        if status_queue is None:
            status_queue = queue.SimpleQueue()

        config_dict = everest_to_ert_config_dict(everest_config)

        runpath_file: Path = Path(
            os.path.join(everest_config.output_dir, ".res_runpath_list")
        )

        assert everest_config.config_file is not None
        config_file: Path = Path(everest_config.config_file)

        model_config = ModelConfig.from_dict(config_dict)

        queue_config = QueueConfig.from_dict(config_dict)
        assert everest_config.simulator is not None
        assert everest_config.simulator.queue_system is not None
        queue_config.queue_options = everest_config.simulator.queue_system
        queue_config.queue_system = everest_config.simulator.queue_system.name

        ensemble_config = get_ensemble_config(config_dict, everest_config)

        substitutions = get_substitutions(
            config_dict, model_config, runpath_file, queue_config.preferred_num_cpu
        )
        ert_templates = read_templates(config_dict)
        _, _, hooked_workflows = workflows_from_dict(config_dict, substitutions)  # type: ignore

        forward_model_steps, env_pr_fm_step = get_forward_model_steps(
            config_dict, substitutions
        )

        env_vars = {}
        for key, val in config_dict.get("SETENV", []):  # type: ignore
            env_vars[key] = substitutions.substitute(val)

        self.support_restart = False
        self._parameter_configuration = ensemble_config.parameter_configuration
        self._parameter_configs = ensemble_config.parameter_configs
        self._response_configuration = ensemble_config.response_configuration

        super().__init__(
            storage,
            runpath_file,
            config_file,
            env_vars,
            env_pr_fm_step,
            model_config,
            queue_config,
            forward_model_steps,
            status_queue,
            substitutions,
            ert_templates,
            hooked_workflows,
            active_realizations=[],  # Set dynamically in run_forward_model()
            log_path=Path(everest_config.optimization_output_dir),
        )

    @classmethod
    def create(
        cls,
        everest_config: EverestConfig,
        simulation_callback: SimulationCallback | None = None,
        optimization_callback: OptimizerCallback | None = None,
        status_queue: queue.SimpleQueue[StatusEvents] | None = None,
    ) -> EverestRunModel:
        return cls(
            everest_config=everest_config,
            simulation_callback=simulation_callback,
            optimization_callback=optimization_callback,
            status_queue=status_queue,
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

        self.ever_storage = EverestStorage(
            output_dir=Path(self._everest_config.optimization_output_dir),
        )
        self.ever_storage.init(self._everest_config)
        self.ever_storage.observe_optimizer(optimizer)

        # Run the optimization:
        optimizer_exit_code = optimizer.run().exit_code

        # Extract the best result from the storage.
        self._result = self.ever_storage.get_optimal_result()

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

    def _init_domain_transforms(
        self, control_variables: NDArray[np.float64]
    ) -> OptModelTransforms:
        model_realizations = self._everest_config.model.realizations
        nreal = len(model_realizations)
        realization_weights = self._everest_config.model.realizations_weights
        if realization_weights is None:
            realization_weights = [1.0 / nreal] * nreal
        transforms = get_optimization_domain_transforms(
            self._everest_config.controls,
            self._everest_config.objective_functions,
            self._everest_config.output_constraints,
            realization_weights,
        )
        # TODO: The transforms are currently needed in the forward model
        # evaluator, but that may be removed later.
        self._domain_transforms = transforms

        # If required, initialize auto-scaling:
        assert isinstance(transforms.objectives, ObjectiveScaler)
        assert transforms.nonlinear_constraints is None or isinstance(
            transforms.nonlinear_constraints, ConstraintScaler
        )
        if transforms.objectives.has_auto_scale or (
            transforms.nonlinear_constraints
            and transforms.nonlinear_constraints.has_auto_scale
        ):
            # Currently the forward models expect scaled variables, so the
            # variable transform must be applied.
            # TODO: when the forward models expect unscaled variables, remove.
            if transforms.variables:
                control_variables = transforms.variables.to_optimizer(control_variables)

            # Run the forward model once to find the objective/constraint values
            # to compute the scales. This will add an ensemble/batch in the
            # storage that is not part of the optimization run. However, the
            # results may be used in the optimization via the caching mechanism.

            self.send_event(
                EverestStatusEvent(
                    batch=None,  # Always 0, but omitting it for consistency
                    everest_event="START_SAMPLING_EVALUATION",
                    exit_code=None,
                )
            )

            objectives, constraints, _ = self._run_forward_model(
                np.repeat(np.expand_dims(control_variables, axis=0), nreal, axis=0),
                model_realizations,
            )

            self.send_event(
                EverestBatchResultEvent(
                    batch=self._batch_id,
                    everest_event="FINISHED_SAMPLING_EVALUATION",
                    result_type="FunctionResult",
                    exit_code=None,
                )
            )

            if transforms.objectives.has_auto_scale:
                transforms.objectives.calculate_auto_scales(objectives)
            if (
                transforms.nonlinear_constraints
                and transforms.nonlinear_constraints.has_auto_scale
            ):
                assert constraints is not None
                transforms.nonlinear_constraints.calculate_auto_scales(constraints)
        return transforms

    def _create_optimizer(self) -> BasicOptimizer:
        # Initialize the optimization model transforms. This may run one initial
        # ensemble for auto-scaling purposes:
        transforms = self._init_domain_transforms(
            np.asarray(
                FlattenedControls(self._everest_config.controls).initial_guesses,
                dtype=np.float64,
            )
        )
        optimizer = BasicOptimizer(
            enopt_config=everest2ropt(self._everest_config, transforms=transforms),
            evaluator=self._forward_model_evaluator,
            transforms=transforms,
        )

        # Before each batch evaluation we check if we should abort:
        optimizer.add_observer(
            EventType.START_EVALUATION,
            functools.partial(
                self._on_before_forward_model_evaluation,
                optimizer=optimizer,
            ),
        )

        def _forward_ropt_event(everest_event: OptimizerEvent) -> None:
            has_results = bool(everest_event.data and everest_event.data.get("results"))

            # The batch these results pertain to
            # If the event has results, they usually pertain to the
            # batch before self._batch_id, i.e., self._batch_id - 1
            exit_code = everest_event.data.get("exit_code")
            exit_code_name = (
                None if exit_code is None else OptimizerExitCode(exit_code).name
            )

            if has_results:
                # A ROPT event may contain multiple results, here we send one
                # event per result
                results = everest_event.data["results"]
                batch_id = results[0].batch_id

                for r in results:
                    self.send_event(
                        EverestBatchResultEvent(
                            batch=batch_id,
                            everest_event=everest_event.event_type.name,
                            result_type=(
                                "FunctionResult"
                                if isinstance(r, FunctionResults)
                                else "GradientResult"
                            ),
                            exit_code=exit_code_name,
                        )
                    )
            else:
                # Events indicating the start of an evaluation,
                # start of optimizer step holds no results
                # but may still be relevant to the subscriber
                self.send_event(
                    EverestStatusEvent(
                        batch=None,
                        everest_event=everest_event.event_type.name,
                        exit_code=exit_code_name,
                    )
                )

        # Forward ROPT events to queue
        for event_type in EventType:
            optimizer.add_observer(event_type, _forward_ropt_event)

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

    def _run_forward_model(
        self,
        control_values: NDArray[np.float64],
        model_realizations: list[int],
        active_control_vectors: list[bool] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None, list[int]]:
        # Reset the current run status:
        self._status = None

        # Get cached_results:
        cached_results = self._get_cached_results(control_values, model_realizations)

        # Collect the indices of the controls that must be evaluated in the batch:
        evaluated_control_indices = [
            idx
            for idx in range(control_values.shape[0])
            if idx not in cached_results
            and (active_control_vectors is None or active_control_vectors[idx])
        ]

        if not evaluated_control_indices:
            cached_results = self._get_cached_results(
                control_values, model_realizations
            )

            objectives = np.zeros(
                (
                    control_values.shape[0],
                    len(self._everest_config.objective_names),
                ),
                dtype=float64,
            )

            constraints = (
                np.zeros(
                    (
                        control_values.shape[0],
                        len(self._everest_config.constraint_names),
                    ),
                    dtype=float64,
                )
                if self._everest_config.constraint_names
                else None
            )

            for control_idx, (
                cached_objectives,
                cached_constraints,
            ) in cached_results.items():
                objectives[control_idx, ...] = cached_objectives
                if constraints is not None:
                    assert cached_constraints is not None
                    constraints[control_idx, ...] = cached_constraints

                # Increase the batch ID for the next evaluation:
            self._batch_id += 1

            # Return the results, together with the indices of the evaluated controls:
            return objectives, constraints, evaluated_control_indices

        # Create the batch to run:
        sim_controls = self._create_simulation_controls(
            control_values, evaluated_control_indices
        )

        # Initialize a new ensemble in storage:
        assert self._experiment is not None
        ensemble = self._experiment.create_ensemble(
            name=f"batch_{self._batch_id}",
            ensemble_size=len(evaluated_control_indices),
            iteration=self._batch_id,
        )
        for sim_id, controls in enumerate(sim_controls.values()):
            self._setup_sim(sim_id, controls, ensemble)

        # Evaluate the batch:
        run_args = self._get_run_args(
            ensemble, model_realizations, evaluated_control_indices
        )
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
        objectives, constraints = self._get_objectives_and_constraints(
            control_values, evaluated_control_indices, results, cached_results
        )

        # Add the results from the evaluations to the cache:
        self._add_results_to_cache(
            control_values,
            model_realizations,
            evaluated_control_indices,
            objectives,
            constraints,
        )

        # Increase the batch ID for the next evaluation:
        self._batch_id += 1

        # Return the results, together with the indices of the evaluated controls:
        return objectives, constraints, evaluated_control_indices

    def _forward_model_evaluator(
        self, control_values: NDArray[np.float64], evaluator_context: EvaluatorContext
    ) -> EvaluatorResult:
        # Currently the forward models expect scaled variables, so the
        # variable transform must be applied.
        # TODO: when the forward models expect unscaled variables, remove. Also
        # the self._transforms attribute can then be removed.
        if self._domain_transforms is not None and self._domain_transforms.variables:
            control_values = self._domain_transforms.variables.to_optimizer(
                control_values
            )

        control_indices = list(range(control_values.shape[0]))
        model_realizations = [
            self._everest_config.model.realizations[evaluator_context.realizations[idx]]
            for idx in control_indices
        ]
        active_control_vectors = [
            evaluator_context.active is None
            or bool(evaluator_context.active[evaluator_context.realizations[idx]])
            for idx in control_indices
        ]
        batch_id = self._batch_id  # Save the batch ID, it will be modified.
        objectives, constraints, evaluated_control_indices = self._run_forward_model(
            control_values, model_realizations, active_control_vectors
        )

        # The simulation id's are a simple enumeration over the evaluated
        # forward models. For the evaluated controls they are therefore
        # implicitly given by there position in the evaluated_control_indices
        # list. We store for each control vector that id, or -1 if it was not
        # evaluated:
        sim_ids = np.fromiter(
            (
                evaluated_control_indices.index(idx)
                if idx in evaluated_control_indices
                else -1
                for idx in control_indices
            ),
            dtype=np.intc,
        )

        return EvaluatorResult(
            objectives=objectives,
            constraints=constraints,
            batch_id=batch_id,
            evaluation_ids=sim_ids,
        )

    def _get_cached_results(
        self, control_values: NDArray[np.float64], model_realizations: list[int]
    ) -> dict[int, Any]:
        cached_results: dict[int, Any] = {}
        if self._simulator_cache is not None:
            for sim_id, model_realization in enumerate(model_realizations):
                cached_data = self._simulator_cache.get(
                    model_realization, control_values[sim_id, :]
                )
                if cached_data is not None:
                    cached_results[sim_id] = cached_data
        return cached_results

    def _create_simulation_controls(
        self,
        control_values: NDArray[np.float64],
        controls_to_evaluate: list[int],
    ) -> dict[int, dict[str, Any]]:
        def _create_control_dicts_for_simulation(
            controls_config: list[ControlConfig], values: NDArray[np.float64]
        ) -> dict[str, Any]:
            control_dicts: dict[str, Any] = {}
            value_list = values.tolist()
            for control in controls_config:
                control_dict: dict[str, Any] = control_dicts.get(control.name, {})
                for variable in control.variables:
                    variable_value = control_dict.get(variable.name, {})
                    if isinstance(variable, ControlVariableGuessListConfig):
                        for index in range(1, len(variable.initial_guess) + 1):
                            variable_value[str(index)] = value_list.pop(0)
                    elif variable.index is not None:
                        variable_value[str(variable.index)] = value_list.pop(0)
                    else:
                        variable_value = value_list.pop(0)
                    control_dict[variable.name] = variable_value
                control_dicts[control.name] = control_dict
            return control_dicts

        return {
            sim_id: _create_control_dicts_for_simulation(
                self._everest_config.controls, control_values[sim_id, :]
            )
            for sim_id in controls_to_evaluate
        }

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
        model_realizations: list[int],
        evaluated_control_indices: list[int],
    ) -> list[RunArg]:
        substitutions = self._substitutions
        self.active_realizations = [True] * len(evaluated_control_indices)
        for sim_id, control_idx in enumerate(evaluated_control_indices):
            substitutions[f"<GEO_ID_{sim_id}_{ensemble.iteration}>"] = str(
                model_realizations[control_idx]
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

    def _get_objectives_and_constraints(
        self,
        control_values: NDArray[np.float64],
        evaluated_control_indices: list[int],
        results: list[dict[str, NDArray[np.float64]]],
        cached_results: dict[int, Any],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
        objectives = self._get_simulation_results(
            results,
            self._everest_config.objective_names,
            control_values,
            evaluated_control_indices,
        )

        constraints = None
        if self._everest_config.output_constraints:
            constraints = self._get_simulation_results(
                results,
                self._everest_config.constraint_names,
                control_values,
                evaluated_control_indices,
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

        return objectives, constraints

    @staticmethod
    def _get_simulation_results(
        results: list[dict[str, NDArray[np.float64]]],
        names: list[str],
        controls: NDArray[np.float64],
        evaluated_control_indices: list[int],
    ) -> NDArray[np.float64]:
        values = np.zeros((controls.shape[0], len(names)), dtype=float64)
        for func_idx, name in enumerate(names):
            values[evaluated_control_indices, func_idx] = np.fromiter(
                (np.nan if not result else result[name][0] for result in results),
                dtype=np.float64,
            )
        return values

    def _add_results_to_cache(
        self,
        control_values: NDArray[np.float64],
        model_realizations: list[int],
        evaluated_control_indices: list[int],
        objectives: NDArray[np.float64],
        constraints: NDArray[np.float64] | None,
    ) -> None:
        if self._simulator_cache is not None:
            for sim_id in evaluated_control_indices:
                self._simulator_cache.add(
                    model_realizations[sim_id],
                    control_values[sim_id, ...],
                    objectives[sim_id, ...],
                    None if constraints is None else constraints[sim_id, ...],
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
        prev_model_realization = None
        jobs: list[JobProgress] = []
        for (
            model_realization,
            simulation,
        ), fm_step in snapshot.get_all_fm_steps().items():
            if model_realization != prev_model_realization:
                prev_model_realization = model_realization
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
                    "realization": model_realization,
                    "simulation": simulation,
                }
            )
            if fm_step.get("error", ""):
                self._handle_errors(
                    batch=self._batch_id,
                    simulation=simulation,
                    model_realization=model_realization,
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
        model_realization: str,
        fm_name: str,
        error_path: str,
        fm_running_err: str,
    ) -> None:
        fm_id = f"b_{batch}_r_{model_realization}_s_{simulation}_{fm_name}"
        fm_logger = logging.getLogger("forward_models")
        if Path(error_path).is_file():
            error_str = Path(error_path).read_text(encoding="utf-8") or fm_running_err
        else:
            error_str = fm_running_err
        error_hash = hash(error_str)
        err_msg = "Batch: {} Realization: {} Simulation: {} Job: {} Failed {}".format(
            batch, model_realization, simulation, fm_name, "\n Error: {} ID:{}"
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
