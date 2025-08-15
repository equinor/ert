from __future__ import annotations

import dataclasses
import datetime
import importlib.metadata
import logging
import os
import queue
import shutil
import traceback
from collections.abc import Callable, MutableSequence
from enum import IntEnum, auto
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import PrivateAttr, ValidationError
from ropt.enums import ExitCode as RoptExitCode
from ropt.evaluator import EvaluatorContext, EvaluatorResult
from ropt.plan import BasicOptimizer
from ropt.results import FunctionResults, Results
from ropt.transforms import OptModelTransforms
from typing_extensions import TypedDict

from ert.config import ParameterConfig, QueueConfig, ResponseConfig
from ert.config.ert_config import (
    create_and_hook_workflows,
    read_templates,
    workflow_jobs_from_dict,
)
from ert.config.model_config import ModelConfig as ErtModelConfig
from ert.ensemble_evaluator import EndEvent, EvaluatorServerConfig
from ert.runpaths import Runpaths
from everest.config import (
    ControlConfig,
    EverestConfig,
    InputConstraintConfig,
    ModelConfig,
    ObjectiveFunctionConfig,
    OptimizationConfig,
    OutputConstraintConfig,
)
from everest.everest_storage import EverestStorage
from everest.optimizer.everest2ropt import everest2ropt
from everest.optimizer.opt_model_transforms import (
    EverestOptModelTransforms,
    get_optimization_domain_transforms,
)
from everest.simulator.everest_to_ert import (
    everest_to_ert_config_dict,
    get_ensemble_config,
    get_forward_model_steps,
    get_substitutions,
    get_workflow_jobs,
)
from everest.strings import EVEREST, STORAGE_DIR

from ..run_arg import RunArg, create_run_arguments
from ..storage.local_ensemble import EverestRealizationInfo
from ..substitutions import Substitutions
from .event import EverestBatchResultEvent, EverestStatusEvent
from .run_model import RunModel, StatusEvents

if TYPE_CHECKING:
    from ert.storage import Ensemble, Experiment


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
    COMPLETED = auto()
    TOO_FEW_REALIZATIONS = auto()
    ALL_REALIZATIONS_FAILED = auto()
    MAX_FUNCTIONS_REACHED = auto()
    MAX_BATCH_NUM_REACHED = auto()
    USER_ABORT = auto()


class _EvaluationStatus(IntEnum):
    TO_SIMULATE = auto()
    INACTIVE = auto()


@dataclasses.dataclass
class _EvaluationInfo:
    control_vector: NDArray[np.float64]
    status: _EvaluationStatus
    flat_index: int
    simulation_id: int | None
    model_realization: int
    perturbation: int
    objectives: NDArray[np.float64] | None = None
    constraints: NDArray[np.float64] | None = None


logger = logging.getLogger(EVEREST)


class EverestRunModel(RunModel):
    optimization_output_dir: str
    simulation_dir: str

    parameter_configuration: list[ParameterConfig]
    response_configuration: list[ResponseConfig]
    ert_templates: list[tuple[str, str]]

    controls: list[ControlConfig]

    objective_functions: list[ObjectiveFunctionConfig]
    objective_names: list[str]

    input_constraints: list[InputConstraintConfig]

    output_constraints: list[OutputConstraintConfig]
    constraint_names: list[str]

    optimization: OptimizationConfig

    model: ModelConfig

    keep_run_path: bool

    _exit_code: EverestExitCode | None = PrivateAttr(default=None)
    _experiment: Experiment | None = PrivateAttr(default=None)
    _eval_server_cfg: EvaluatorServerConfig | None = PrivateAttr(default=None)
    _batch_id: int = PrivateAttr(default=0)
    _ever_storage: EverestStorage | None = PrivateAttr(default=None)
    _opt_callback: OptimizerCallback | None = PrivateAttr(default=None)

    _transforms: EverestOptModelTransforms = PrivateAttr()

    def __init__(self, **data: Any) -> None:
        transforms = data.pop("transforms", None)
        opt_callback = data.pop("opt_callback", None)
        super().__init__(**data)
        self._transforms = transforms
        self._opt_callback = opt_callback

    @classmethod
    def create(
        cls,
        everest_config: EverestConfig,
        optimization_callback: OptimizerCallback | None = None,
        status_queue: queue.SimpleQueue[StatusEvents] | None = None,
    ) -> EverestRunModel:
        logger.info(
            "Using random seed: %d. To deterministically reproduce this experiment, "
            "add the above random seed to your configuration file.",
            everest_config.environment.random_seed,
        )

        storage_dir = os.path.join(everest_config.output_dir, STORAGE_DIR)

        if status_queue is None:
            status_queue = queue.SimpleQueue()

        config_dict = everest_to_ert_config_dict(everest_config)

        runpath_file: Path = Path(
            os.path.join(everest_config.output_dir, ".res_runpath_list")
        )

        assert everest_config.config_file is not None
        config_file: Path = Path(everest_config.config_path)

        runpath_config = ErtModelConfig.from_dict(config_dict)

        queue_config = QueueConfig.from_dict(config_dict)
        assert everest_config.simulator is not None
        assert everest_config.simulator.queue_system is not None
        queue_config.queue_options = everest_config.simulator.queue_system
        queue_config.queue_system = everest_config.simulator.queue_system.name

        ensemble_config = get_ensemble_config(config_dict, everest_config)

        substitutions = get_substitutions(
            config_dict,
            runpath_config,
            runpath_file,
            queue_config.queue_options.num_cpu,
        )
        ert_templates = read_templates(config_dict)

        workflow_jobs = get_workflow_jobs(everest_config)
        if deprecated_workflow_jobs := workflow_jobs_from_dict(config_dict):
            workflow_jobs.update(deprecated_workflow_jobs)
        _, hooked_workflows = create_and_hook_workflows(
            config_dict, workflow_jobs, substitutions
        )

        forward_model_steps, env_pr_fm_step = get_forward_model_steps(
            everest_config, config_dict, substitutions
        )

        env_vars = {}
        substituter = Substitutions(substitutions)
        for key, val in config_dict.get("SETENV", []):  # type: ignore
            env_vars[key] = substituter.substitute(val)

        transforms: EverestOptModelTransforms = get_optimization_domain_transforms(
            everest_config.controls,
            everest_config.objective_functions,
            everest_config.output_constraints,
            everest_config.model,
        )

        delete_run_path: bool = (
            everest_config.simulator is not None
            and everest_config.simulator.delete_run_path
        )

        return cls(
            controls=everest_config.controls,
            simulation_dir=everest_config.simulation_dir,
            keep_run_path=not delete_run_path,
            objective_names=everest_config.objective_names,
            constraint_names=everest_config.constraint_names,
            objective_functions=everest_config.objective_functions,
            input_constraints=everest_config.input_constraints,
            output_constraints=everest_config.output_constraints,
            optimization=everest_config.optimization,
            model=everest_config.model,
            transforms=transforms,
            optimization_output_dir=everest_config.optimization_output_dir,
            log_path=everest_config.log_dir,
            random_seed=everest_config.environment.random_seed,
            runpath_file=runpath_file,
            # Mutated throughout execution of Everest
            # (Not totally in conformity with ERT runmodel logic)
            active_realizations=[],
            parameter_configuration=ensemble_config.parameter_configuration,
            response_configuration=ensemble_config.response_configuration,
            ert_templates=ert_templates,
            user_config_file=config_file,
            env_vars=env_vars,
            env_pr_fm_step=env_pr_fm_step,
            runpath_config=runpath_config,
            forward_model_steps=forward_model_steps,
            substitutions=substitutions,
            hooked_workflows=hooked_workflows,
            storage_path=storage_dir,
            queue_config=queue_config,
            status_queue=status_queue,
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

    def __repr__(self) -> str:
        return f"EverestRunModel(config={self.user_config_file})"

    def start_simulations_thread(
        self,
        evaluator_server_config: EvaluatorServerConfig,
        rerun_failed_realizations: bool = False,
    ) -> None:
        failed = False
        exception: Exception | None = None
        error_messages: MutableSequence[str] = []
        traceback_str: str | None = None
        try:
            logger.debug("Starting Everest simulations thread")
            self.run_experiment(evaluator_server_config)
        except Exception as e:
            failed = True
            exception = e
            traceback_str = traceback.format_exc()
            logger.error(f"Experiment failed with exception:\n{traceback_str}")
            raise
        finally:
            if self._exit_code not in {
                EverestExitCode.COMPLETED,
                EverestExitCode.MAX_FUNCTIONS_REACHED,
                EverestExitCode.MAX_BATCH_NUM_REACHED,
            }:
                failed = True
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

    def _handle_optimizer_results(self, results: tuple[Results, ...]) -> None:
        assert self._ever_storage is not None
        self._ever_storage.on_batch_evaluation_finished(results)

        for r in results:
            storage_batches = (
                self._ever_storage.data.batches_with_function_results
                if isinstance(r, FunctionResults)
                else self._ever_storage.data.batches_with_gradient_results
            )
            batch_data = next(
                (b for b in storage_batches if b.batch_id == r.batch_id),
                None,
            )

            self.send_event(
                EverestBatchResultEvent(
                    batch=r.batch_id,
                    everest_event="OPTIMIZATION_RESULT",
                    result_type="FunctionResult"
                    if isinstance(r, FunctionResults)
                    else "GradientResult",
                    results=batch_data.to_dict() if batch_data else None,
                )
            )

    def run_experiment(
        self,
        evaluator_server_config: EvaluatorServerConfig,
        rerun_failed_realizations: bool = False,
    ) -> None:
        self.log_at_startup()
        self._eval_server_cfg = evaluator_server_config

        self._experiment = self._experiment or self._storage.create_experiment(
            name=f"EnOpt@{datetime.datetime.now().isoformat(timespec='seconds')}",
            parameters=self.parameter_configuration,
            responses=self.response_configuration,
        )

        # Initialize the ropt optimizer:
        optimizer, initial_guesses = self._create_optimizer()

        self._ever_storage = EverestStorage(
            output_dir=Path(self.optimization_output_dir),
        )

        formatted_control_names = [
            name for config in self.controls for name in config.formatted_control_names
        ]
        self._ever_storage.init(
            formatted_control_names=formatted_control_names,
            objective_functions=self.objective_functions,
            output_constraints=self.output_constraints,
            realizations=self.model.realizations,
        )
        optimizer.set_results_callback(self._handle_optimizer_results)

        # Run the optimization:
        optimizer_exit_code = optimizer.run(initial_guesses).exit_code

        # Store some final results.
        self._ever_storage.on_optimization_finished()
        if (
            optimizer_exit_code is not RoptExitCode.UNKNOWN
            and optimizer_exit_code is not RoptExitCode.TOO_FEW_REALIZATIONS
            and optimizer_exit_code is not RoptExitCode.USER_ABORT
        ):
            self._ever_storage.export_everest_opt_results_to_csv()

        if self._exit_code is None:
            match optimizer_exit_code:
                case RoptExitCode.MAX_FUNCTIONS_REACHED:
                    self._exit_code = EverestExitCode.MAX_FUNCTIONS_REACHED
                case RoptExitCode.MAX_BATCHES_REACHED:
                    self._exit_code = EverestExitCode.MAX_BATCH_NUM_REACHED
                case RoptExitCode.USER_ABORT:
                    self._exit_code = EverestExitCode.USER_ABORT
                case RoptExitCode.TOO_FEW_REALIZATIONS:
                    self._exit_code = (
                        EverestExitCode.TOO_FEW_REALIZATIONS
                        if self.get_number_of_successful_realizations() > 0
                        else EverestExitCode.ALL_REALIZATIONS_FAILED
                    )
                case _:
                    self._exit_code = EverestExitCode.COMPLETED

        logger.debug(
            f"Everest experiment finished with exit code {self._exit_code.name}"
        )

    def _check_for_abort(self) -> bool:
        logger.debug("Optimization callback called")
        if (
            self._opt_callback is not None
            and self._opt_callback() == "stop_optimization"
        ):
            logger.info("User abort requested.")
            return True
        return False

    def _create_optimizer(self) -> tuple[BasicOptimizer, list[float]]:
        enopt_config, initial_guesses = everest2ropt(
            self.controls,
            self.objective_functions,
            self.input_constraints,
            self.output_constraints,
            self.optimization,
            self.model,
            self.random_seed,
            self.optimization_output_dir,
        )
        transforms = (
            OptModelTransforms(
                variables=self._transforms["control_scaler"],
                objectives=self._transforms["objective_scaler"],
                nonlinear_constraints=self._transforms["constraint_scaler"],
            )
            if self._transforms
            else None
        )
        try:
            optimizer = BasicOptimizer(
                enopt_config=enopt_config,
                transforms=transforms,
                evaluator=self._forward_model_evaluator,
            )
        except ValidationError as exc:
            ert_version = importlib.metadata.version("ert")
            ropt_version = importlib.metadata.version("ropt")
            msg = (
                f"Validation error(s) in ropt:\n\n{exc}.\n\n"
                "Check the everest installation, there may a be version mismatch.\n"
                f"  (ERT: {ert_version}, ropt: {ropt_version})\n"
                "If the everest installation is correct, please report this as a bug."
            )
            raise ValueError(msg) from exc

        # Before each batch evaluation we check if we should abort:
        optimizer.set_abort_callback(self._check_for_abort)

        return optimizer, initial_guesses

    def _run_forward_model(
        self,
        sim_to_control_vector: NDArray[np.float64],
        sim_to_model_realization: list[int],
        sim_to_perturbation: list[int],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
        # Initialize a new ensemble in storage:
        assert self._experiment is not None
        ensemble = self._experiment.create_ensemble(
            name=f"batch_{self._batch_id}",
            ensemble_size=sim_to_control_vector.shape[0],
            iteration=self._batch_id,
        )

        realization_info: dict[int, EverestRealizationInfo] = {
            ert_realization: {
                "model_realization": model_realization,
                "perturbation": perturbation,
            }
            for ert_realization, (model_realization, perturbation) in enumerate(
                zip(
                    sim_to_model_realization,
                    sim_to_perturbation,
                    strict=False,
                )
            )
        }

        ensemble.save_everest_realization_info(realization_info)

        for sim_id in range(sim_to_control_vector.shape[0]):
            sim_controls = sim_to_control_vector[sim_id]
            offset = 0
            for control_config in self.controls:
                ext_param_config = next(
                    c
                    for c in self.parameter_configuration
                    if c.name == control_config.name
                )
                n_param_keys = len(ext_param_config.parameter_keys)

                # Save controls to ensemble
                ensemble.save_parameters_numpy(
                    sim_controls[offset : (offset + n_param_keys)].reshape(-1, 1),
                    ext_param_config.name,
                    np.array([sim_id]),
                )
                offset += n_param_keys

        # Evaluate the batch:
        run_args = self._get_run_args(ensemble, sim_to_model_realization)
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
        self._delete_run_path(run_args)

        # Gather the results
        objectives, constraints = self._gather_simulation_results(ensemble)

        # Return the results, together with the indices of the evaluated controls:
        return objectives, constraints

    def _forward_model_evaluator(
        self, control_values: NDArray[np.float64], evaluator_context: EvaluatorContext
    ) -> EvaluatorResult:
        logger.debug(f"Evaluating batch {self._batch_id}")

        # ----------------------------------------------------------------------
        # General Info:
        #
        # `control_values` is a matrix, where each row is one set of controls,
        # One forward model run must be done for each row, but only if the
        # corresponding model realization is marked as active, as indicated in
        # the `evaluator_context` object.
        #
        # The result consists of matrices for objectives and constraints. Each
        # row corresponds to a row in `control_values` and contains the results
        # of the corresponding forward model run.
        #
        # Following information is used from `evaluator_context`:
        #
        # 1. `evaluator_context.realizations`: The indices of the model
        #    realizations for each control vector. A numpy vector with a length
        #    equal to the number of rows of `control_values`
        # 2. `evaluator_context.perturbations`: The indices of the perturbations
        #    for each control vector. A numpy vector with a length equal to the
        #    number of rows of `control_values`. If an entry is less than zero,
        #    the corresponding control vector is not a perturbation. If
        #    evaluator_context.perturbations is `None`, none of the vectors is a
        #    perturbation.
        #
        # Control vectors pertaining to inactive realizations do not need to be
        # evaluated. This can be achieved by extracting active entries before
        # running the forward models, using the
        # `evaluator_context.get_active_evaluations` method of the context
        # object. Before returning the results, the must be amended by inserting
        # rows at the positions that were filtered out. This can be done using
        # the `evaluator_context.insert_inactive_results`
        #
        # In summary, the evaluation comprises three steps:
        #
        # 1. A filter step, where all inactive control vectors are removed.
        # 2. A forward model run for each remaining control vector.
        # 3. A reconstruction step where zero values are inserted in the results
        #    for inactive control vectors.
        #
        # Note: An extra step may inserted before the last step in one of the
        #       initial batches, where auto-scaling values are calculated. This
        #       is done at that point for efficiency reasons, but has nothing to
        #       do with the forward model evaluations itself.
        #
        # ----------------------------------------------------------------------

        # This is the first step: Remove inactive control vectors.
        #
        # This generates the following vectors that have the necessary information
        # to run the forward models for all active control vectors:
        #
        # 1. active_control_vectors: A copy of the `control_values` matrix, where
        #    all inactive control vectors have been removed.
        # 2. `realization_indices` and `perturbation_indices` are copies of
        #    `evaluator_context.realizations` and
        #    `evaluator_context.perturbations` with entries corresponding to
        #    inactive control vectors removed.
        active_control_vectors = evaluator_context.get_active_evaluations(
            control_values
        )
        num_simulations = active_control_vectors.shape[0]
        realization_indices = evaluator_context.get_active_evaluations(
            evaluator_context.realizations
        )
        perturbation_indices = (
            np.full(num_simulations, fill_value=-1, dtype=np.intc)
            if evaluator_context.perturbations is None
            else evaluator_context.get_active_evaluations(
                evaluator_context.perturbations
            )
        )

        if num_simulations > 0:
            self.send_event(
                EverestStatusEvent(
                    batch=self._batch_id, everest_event="START_OPTIMIZER_EVALUATION"
                )
            )

            # Run the forward model and collect the objectives and constraints:
            logger.debug(f"Running forward model for batch {self._batch_id}")

            # Find the model realization name of each active control vector, by
            # finding its realization index and then looking up its name in the
            # config:
            model_realizations = [
                self.model.realizations[realization_indices[idx]]
                for idx in range(num_simulations)
            ]

            # Run the forward models:
            objectives, constraints = self._run_forward_model(
                sim_to_control_vector=active_control_vectors,
                sim_to_model_realization=model_realizations,
                sim_to_perturbation=perturbation_indices.tolist(),
            )

            self.send_event(
                EverestStatusEvent(
                    batch=self._batch_id,
                    everest_event="FINISHED_OPTIMIZER_EVALUATION",
                )
            )

            # The simulation IDs are also returned, these are implicitly
            # defined as the range over the active control vectors:
            sim_ids: NDArray[np.int32] = np.arange(num_simulations, dtype=np.int32)

            # Calculate auto-scales if necessary.
            self._calculate_objective_auto_scales(
                objectives, realization_indices, perturbation_indices
            )
            if constraints is not None:
                self._calculate_constraint_auto_scales(
                    constraints, realization_indices, perturbation_indices
                )

            # This is the final step: insert zero results for inactive
            # control vectors. This is done by inserting zeros at each position
            # where the input control vectors are not active.
            objectives = evaluator_context.insert_inactive_results(objectives)
            if constraints is not None:
                constraints = evaluator_context.insert_inactive_results(constraints)
            sim_ids = evaluator_context.insert_inactive_results(sim_ids, fill_value=-1)
        else:
            # Nothing to do, there may only have been inactive control vectors:
            num_all_simulations = control_values.shape[0]
            objectives = np.zeros(
                (num_all_simulations, len(self.objective_names)),
                dtype=np.float64,
            )
            constraints = (
                np.zeros(
                    (num_all_simulations, len(self.constraint_names)),
                    dtype=np.float64,
                )
                if self.output_constraints
                else None
            )
            sim_ids = np.array([-1] * num_all_simulations, dtype=np.int32)

        evaluator_result = EvaluatorResult(
            objectives=objectives,
            constraints=constraints,
            batch_id=self._batch_id,
            evaluation_info={"sim_ids": sim_ids},
        )

        # increase the batch ID for the next evaluation:
        self._batch_id += 1

        return evaluator_result

    def _calculate_objective_auto_scales(
        self,
        objectives: NDArray[np.float64],
        realization_indices: NDArray[np.intc],
        perturbation_indices: NDArray[np.intc],
    ) -> None:
        objective_transform = self._transforms["objective_scaler"]
        if objective_transform.needs_auto_scale_calculation:
            mask = perturbation_indices < 0
            if not np.any(mask):  # If we have only perturbations, just use those.
                mask = np.ones(perturbation_indices.shape[0], dtype=np.bool_)
            objective_transform.calculate_auto_scales(
                objectives[mask, :], realization_indices[mask]
            )

    def _calculate_constraint_auto_scales(
        self,
        constraints: NDArray[np.float64],
        realization_indices: NDArray[np.intc],
        perturbation_indices: NDArray[np.intc],
    ) -> None:
        constraint_transform = self._transforms["constraint_scaler"]
        assert constraint_transform is not None
        if constraint_transform.needs_auto_scale_calculation:
            mask = perturbation_indices < 0
            if not np.any(mask):  # If we have only perturbations, just use those.
                mask = np.ones(perturbation_indices.shape[0], dtype=np.bool_)
            constraint_transform.calculate_auto_scales(
                constraints[mask, :], realization_indices[mask]
            )

    def _get_run_args(
        self,
        ensemble: Ensemble,
        sim_to_model_realization: list[int],
    ) -> list[RunArg]:
        substitutions = self.substitutions
        self.active_realizations = [True] * len(sim_to_model_realization)
        for sim_id, model_realization in enumerate(sim_to_model_realization):
            substitutions[f"<GEO_ID_{sim_id}_{ensemble.iteration}>"] = str(
                int(model_realization)
            )
        run_paths = Runpaths(
            jobname_format=self.runpath_config.jobname_format_string,
            runpath_format=self.runpath_config.runpath_format_string,
            filename=str(self.runpath_file),
            substitutions=substitutions,
            eclbase=self.runpath_config.eclbase_format_string,
        )
        return create_run_arguments(
            run_paths,
            self.active_realizations,
            ensemble=ensemble,
        )

    def _delete_run_path(self, run_args: list[RunArg]) -> None:
        logger.debug("Simulation callback called")
        if not self.keep_run_path:
            for i, real in self.get_current_snapshot().reals.items():
                path_to_delete = run_args[int(i)].runpath
                if real.get("status") == "Finished" and os.path.isdir(path_to_delete):

                    def onerror(
                        _: Callable[..., Any],
                        path: str,
                        sys_info: tuple[
                            type[BaseException], BaseException, TracebackType
                        ],
                    ) -> None:
                        logger.debug(f"Failed to remove {path}, {sys_info}")

                    shutil.rmtree(path_to_delete, onerror=onerror)

    def _gather_simulation_results(
        self, ensemble: Ensemble
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
        objective_names = self.objective_names
        objectives = np.zeros((ensemble.ensemble_size, len(objective_names)))

        constraint_names = self.constraint_names
        constraints = np.zeros((ensemble.ensemble_size, len(constraint_names)))

        if not any(self.active_realizations):
            nan_objectives = np.full(
                (ensemble.ensemble_size, len(objective_names)), fill_value=np.nan
            )
            nan_constraints = (
                np.full(
                    (ensemble.ensemble_size, len(constraint_names)), fill_value=np.nan
                )
                if constraint_names
                else None
            )
            return nan_objectives, nan_constraints

        for sim_id, successful in enumerate(self.active_realizations):
            if not successful:
                logger.error(f"Simulation {sim_id} failed.")
                objectives[sim_id, :] = np.nan
                constraints[sim_id, :] = np.nan
                continue

            for i, obj_name in enumerate(objective_names):
                data = ensemble.load_responses(obj_name, (sim_id,))
                objectives[sim_id, i] = data["values"].item()

            for i, constr_name in enumerate(constraint_names):
                data = ensemble.load_responses(constr_name, (sim_id,))
                constraints[sim_id, i] = data["values"].item()

        return objectives, constraints if constraint_names else None

    def check_if_runpath_exists(self) -> bool:
        return (
            self.simulation_dir is not None
            and os.path.exists(self.simulation_dir)
            and any(os.listdir(self.simulation_dir))
        )
