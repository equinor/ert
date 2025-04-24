from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import os
import queue
import shutil
import traceback
from collections.abc import Callable, MutableSequence
from enum import IntEnum, auto
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import polars as pl
from numpy._typing import NDArray
from ropt.enums import OptimizerExitCode
from ropt.evaluator import EvaluatorContext, EvaluatorResult
from ropt.plan import BasicOptimizer
from ropt.results import FunctionResults, Results
from ropt.transforms import OptModelTransforms
from typing_extensions import TypedDict

from ert.config import ExtParamConfig
from ert.config.ert_config import (
    create_and_hook_workflows,
    read_templates,
    workflow_jobs_from_dict,
)
from ert.config.model_config import ModelConfig
from ert.config.queue_config import QueueConfig
from ert.ensemble_evaluator import EndEvent, EvaluatorServerConfig
from ert.runpaths import Runpaths
from ert.storage import open_storage
from everest.config import ControlConfig, ControlVariableGuessListConfig, EverestConfig
from everest.config.utils import FlattenedControls
from everest.everest_storage import (
    EverestStorage,
    OptimalResult,
)
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
    get_workflow_jobs,
)
from everest.strings import EVEREST, STORAGE_DIR

from ..run_arg import RunArg, create_run_arguments
from ..storage.local_ensemble import EverestRealizationInfo
from .base_run_model import BaseRunModel, StatusEvents
from .event import (
    EverestBatchResultEvent,
    EverestCacheHitEvent,
    EverestStatusEvent,
)

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
    COMPLETED = 1
    TOO_FEW_REALIZATIONS = 2
    MAX_FUNCTIONS_REACHED = 3
    MAX_BATCH_NUM_REACHED = 4
    USER_ABORT = 5


class _EvaluationStatus(IntEnum):
    TO_SIMULATE = auto()
    CACHED = auto()
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


class EverestRunModel(BaseRunModel):
    def __init__(
        self,
        everest_config: EverestConfig,
        optimization_callback: OptimizerCallback | None,
        status_queue: queue.SimpleQueue[StatusEvents] | None = None,
    ):
        Path(everest_config.log_dir).mkdir(parents=True, exist_ok=True)
        Path(everest_config.optimization_output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            "Using random seed: %d. To deterministically reproduce this experiment, "
            "add the above random seed to your configuration file.",
            everest_config.environment.random_seed,
        )

        self._everest_config = everest_config
        self._opt_callback = optimization_callback
        self._fm_errors: dict[int, dict[str, Any]] = {}
        self._result: OptimalResult | None = None
        self._exit_code: EverestExitCode | None = None
        self._experiment: Experiment | None = None
        self._eval_server_cfg: EvaluatorServerConfig | None = None
        self._batch_id: int = 0

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
        for key, val in config_dict.get("SETENV", []):  # type: ignore
            env_vars[key] = substitutions.substitute(val)

        self.support_restart = False
        self._parameter_configuration = ensemble_config.parameter_configuration
        self._parameter_configs = ensemble_config.parameter_configs
        self._response_configuration = ensemble_config.response_configuration
        self._templates = ert_templates

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
            hooked_workflows,
            random_seed=123,  # No-op as far as Everest is concerned
            active_realizations=[],  # Set dynamically in run_forward_model()
            log_path=Path(everest_config.optimization_output_dir),
        )

    @classmethod
    def create(
        cls,
        everest_config: EverestConfig,
        optimization_callback: OptimizerCallback | None = None,
        status_queue: queue.SimpleQueue[StatusEvents] | None = None,
    ) -> EverestRunModel:
        return cls(
            everest_config=everest_config,
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

    def start_simulations_thread(
        self, evaluator_server_config: EvaluatorServerConfig, restart: bool = False
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
        self.ever_storage.on_batch_evaluation_finished(results)

        for r in results:
            storage_batches = (
                self.ever_storage.data.batches_with_function_results
                if isinstance(r, FunctionResults)
                else self.ever_storage.data.batches_with_gradient_results
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
        self, evaluator_server_config: EvaluatorServerConfig, restart: bool = False
    ) -> None:
        self.log_at_startup()
        self._eval_server_cfg = evaluator_server_config

        self._experiment = self._experiment or self._storage.create_experiment(
            name=f"EnOpt@{datetime.datetime.now().strftime('%Y-%m-%d@%H:%M:%S')}",
            parameters=self._parameter_configuration,
            responses=self._response_configuration,
        )

        # Initialize the ropt optimizer:
        optimizer = self._create_optimizer()

        self.ever_storage = EverestStorage(
            output_dir=Path(self._everest_config.optimization_output_dir),
        )

        self.ever_storage.init(
            formatted_control_names=self._everest_config.formatted_control_names,
            objective_functions=self._everest_config.objective_functions,
            output_constraints=self._everest_config.output_constraints,
            realizations=self._everest_config.model.realizations,
        )
        optimizer.set_results_callback(self._handle_optimizer_results)

        # Run the optimization:
        optimizer_exit_code = optimizer.run().exit_code

        # Store some final results.
        self.ever_storage.on_optimization_finished()

        # Extract the best result from the storage.
        self._result = self.ever_storage.get_optimal_result()

        if self._exit_code is None:
            match optimizer_exit_code:
                case OptimizerExitCode.MAX_FUNCTIONS_REACHED:
                    self._exit_code = EverestExitCode.MAX_FUNCTIONS_REACHED
                case OptimizerExitCode.MAX_BATCHES_REACHED:
                    self._exit_code = EverestExitCode.MAX_BATCH_NUM_REACHED
                case OptimizerExitCode.USER_ABORT:
                    self._exit_code = EverestExitCode.USER_ABORT
                case OptimizerExitCode.TOO_FEW_REALIZATIONS:
                    self._exit_code = EverestExitCode.TOO_FEW_REALIZATIONS
                case _:
                    self._exit_code = EverestExitCode.COMPLETED

        logger.debug(
            f"Everest experiment finished with exit code {self._exit_code.name}"
        )

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

        # If required, initialize auto-scaling:
        assert isinstance(transforms.objectives, ObjectiveScaler)
        assert transforms.nonlinear_constraints is None or isinstance(
            transforms.nonlinear_constraints, ConstraintScaler
        )
        if transforms.objectives.has_auto_scale or (
            transforms.nonlinear_constraints
            and transforms.nonlinear_constraints.has_auto_scale
        ):
            # Run the forward model once to find the objective/constraint values
            # to compute the scales. This will add an ensemble/batch in the
            # storage that is not part of the optimization run. However, the
            # results may be used in the optimization via the caching mechanism.

            self.send_event(
                EverestStatusEvent(
                    batch=self._batch_id,
                    everest_event="START_SAMPLING_EVALUATION",
                )
            )

            logger.debug(f"Running sampling forward model for batch {self._batch_id}")
            objectives, constraints = self._run_forward_model(
                sim_to_control_vector=np.repeat(
                    np.expand_dims(control_variables, axis=0), nreal, axis=0
                ),
                sim_to_model_realization=model_realizations,
                sim_to_perturbation=[-1] * nreal,
            )

            self.send_event(
                EverestBatchResultEvent(
                    batch=self._batch_id,
                    everest_event="FINISHED_SAMPLING_EVALUATION",
                    result_type="FunctionResult",
                )
            )

            # Increase the batch ID for the next evaluation:
            self._batch_id += 1

            if transforms.objectives.has_auto_scale:
                transforms.objectives.calculate_auto_scales(objectives)
            if (
                transforms.nonlinear_constraints
                and transforms.nonlinear_constraints.has_auto_scale
            ):
                assert constraints is not None
                transforms.nonlinear_constraints.calculate_auto_scales(constraints)
        return transforms

    def _check_for_abort(self) -> bool:
        logger.debug("Optimization callback called")
        if (
            self._opt_callback is not None
            and self._opt_callback() == "stop_optimization"
        ):
            logger.info("User abort requested.")
            return True
        return False

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
            everest_config=self._everest_config,
        )

        # Before each batch evaluation we check if we should abort:
        optimizer.set_abort_callback(self._check_for_abort)

        return optimizer

    def _run_forward_model(
        self,
        sim_to_control_vector: NDArray[np.float64],
        sim_to_model_realization: list[int],
        sim_to_perturbation: list[int],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
        # Reset the current run status:

        # Create the batch to run:
        sim_controls = self._create_simulation_controls(sim_to_control_vector)

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

        for sim_id, controls in enumerate(sim_controls.values()):
            self._setup_sim(sim_id, controls, ensemble)

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
        self._delete_runpath(run_args)

        # Gather the results
        objectives, constraints = self._gather_simulation_results(ensemble)

        # Return the results, together with the indices of the evaluated controls:
        return objectives, constraints

    @staticmethod
    def find_cached_results(
        control_values: NDArray[np.float64],
        model_realizations: list[int],
        all_results: pl.DataFrame | None,
        control_names: list[str],
    ) -> pl.DataFrame | None:
        if all_results is None:
            return None
        # Will be used to search the cache
        controls_to_evaluate_df = pl.DataFrame(
            {
                "flat_index": list(range(len(model_realizations))),
                "model_realization": pl.Series(model_realizations, dtype=pl.UInt16),
                **{
                    control_name: pl.Series(control_values[:, i], dtype=pl.Float64)
                    for i, control_name in enumerate(control_names)
                },
            }
        )

        EPS = float(np.finfo(np.float32).eps)

        left = all_results
        right = controls_to_evaluate_df

        # Note: asof join will approximate for floats, but
        # only for one column at a time, this for loop
        # will incrementally do the join, filtering out
        # mismatching rows iteratively. If in the future, the `on` argument
        # accepts multiple columns, we can drop this for loop.
        # One control-value-column at a time,
        # we filter out rows with control values
        # that are not approximately matching the control value
        # of our to-be-evaluated controls.
        for control_name in control_names:
            if "flat_index" in left.columns:
                left = left.drop("flat_index")

            left = left.sort(["model_realization", control_name]).join_asof(
                right.sort(["model_realization", control_name]),
                on=control_name,
                by="model_realization",  # pre-join by model realization
                tolerance=EPS,  # Same as np.allclose with atol=EPS
                strategy="nearest",
                check_sortedness=False,
                # Ref: https://github.com/pola-rs/polars/issues/21693
            )

            left = left.filter(pl.col("flat_index").is_not_null())

            left = left.drop([s for s in left.columns if s.endswith("_right")])

            if left.is_empty():
                break

        return (
            left.rename({"realization": "simulation_id"}).drop(
                "model_realization", "perturbation"
            )
            if not left.is_empty()
            else None
        )

    @staticmethod
    def _create_evaluation_infos(
        control_values: NDArray[np.float64],
        model_realizations: list[int],
        perturbations: list[int],
        active_controls: list[bool],
        all_results: pl.DataFrame | None,
        control_names: list[str],
        objective_names: list[str],
        constraint_names: list[str],
    ) -> tuple[list[_EvaluationInfo], pl.DataFrame | None]:
        inactive_objective_fill_value: NDArray[np.float64] = np.zeros(
            len(objective_names)
        )
        inactive_constraints_fill_value: NDArray[np.float64] = np.zeros(
            len(constraint_names)
        )
        evaluation_infos = []

        cache_hits_df = EverestRunModel.find_cached_results(
            control_values, model_realizations, all_results, control_names
        )

        sim_id_counter = 0
        for flat_index, (
            control_row,
            model_realization,
            perturbation,
            is_active,
        ) in enumerate(
            zip(
                control_values,
                model_realizations,
                perturbations,
                active_controls,
                strict=False,
            )
        ):
            control_vector = cast(NDArray[np.float64], control_row)
            if not is_active:
                evaluation_infos.append(
                    _EvaluationInfo(
                        control_vector=control_vector,
                        status=_EvaluationStatus.INACTIVE,
                        model_realization=model_realization,
                        perturbation=perturbation,
                        flat_index=flat_index,
                        simulation_id=None,
                        objectives=inactive_objective_fill_value,
                        constraints=inactive_constraints_fill_value,
                    )
                )
                continue

            if cache_hits_df is not None and not cache_hits_df.is_empty():
                hit_row = cache_hits_df.filter(
                    cache_hits_df["flat_index"] == flat_index
                )

                if not hit_row.is_empty():
                    # Cache hit!
                    row_dict = hit_row.to_dicts()[0]

                    objectives = np.array(
                        [row_dict[o] for o in objective_names], dtype=np.float64
                    )
                    constraints = np.array(
                        [row_dict[c] for c in constraint_names], dtype=np.float64
                    )

                    evaluation_infos.append(
                        _EvaluationInfo(
                            control_vector=control_vector,
                            status=_EvaluationStatus.CACHED,
                            model_realization=model_realization,
                            perturbation=perturbation,
                            flat_index=flat_index,
                            simulation_id=None,
                            objectives=objectives,
                            constraints=constraints,
                        )
                    )
                    continue

            evaluation_infos.append(
                _EvaluationInfo(
                    control_vector=control_vector,
                    status=_EvaluationStatus.TO_SIMULATE,
                    model_realization=model_realization,
                    perturbation=perturbation,
                    flat_index=flat_index,
                    simulation_id=sim_id_counter,
                )
            )
            sim_id_counter += 1

        cache_hits = (
            cache_hits_df[["batch", "simulation_id", "flat_index"]]
            if cache_hits_df is not None and not cache_hits_df.is_empty()
            else None
        )
        return evaluation_infos, cache_hits

    def _forward_model_evaluator(
        self, control_values: NDArray[np.float64], evaluator_context: EvaluatorContext
    ) -> EvaluatorResult:
        logger.debug(f"Evaluating batch {self._batch_id}")
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

        num_constraints = len(self._everest_config.constraint_names)

        assert self._experiment is not None
        all_results = self._experiment.all_parameters_and_gen_data

        evaluation_infos, cache_hits_df = self._create_evaluation_infos(
            control_values=control_values,
            model_realizations=model_realizations,
            perturbations=evaluator_context.perturbations.tolist()
            if evaluator_context.perturbations is not None
            else [-1] * len(model_realizations),
            active_controls=active_control_vectors,
            control_names=self._everest_config.formatted_control_names,
            objective_names=self._everest_config.objective_names,
            constraint_names=self._everest_config.constraint_names,
            all_results=all_results,
        )

        cache_hits_dict: list[dict[str, Any]] = []
        if cache_hits_df is not None and not cache_hits_df.is_empty():
            perturbations = (
                evaluator_context.perturbations
                if evaluator_context.perturbations is not None
                else [-1] * len(model_realizations)
            )
            cache_hits_dict = (
                cache_hits_df.with_columns(
                    pl.col("flat_index")
                    .map_elements(
                        # Assume none of the evals are perturbations if there is no
                        # evaluator_context.perturbations
                        lambda i: perturbations[i],
                        return_dtype=pl.Int32,
                    )
                    .alias("target_perturbation"),
                    pl.col("flat_index")
                    .map_elements(
                        lambda i: evaluator_context.realizations[i],
                        return_dtype=pl.Int32,
                    )
                    .alias("model_realization"),  # Same for source and target
                )
                .rename(
                    {
                        "batch": "source_batch_id",
                        "simulation_id": "source_simulation_id",
                        "flat_index": "target_evaluation_id",
                    }
                )
                .to_dicts()
            )

            logger.debug(f"Found {len(cache_hits_dict)} cache_hits: {cache_hits_dict}")
            self.send_event(
                EverestCacheHitEvent(batch=self._batch_id, data=cache_hits_dict)
            )

        control_values_to_simulate = np.array(
            [
                c.control_vector
                for c in evaluation_infos
                if c.status == _EvaluationStatus.TO_SIMULATE
            ],
            dtype=np.float64,
        )

        if control_values_to_simulate.shape[0] > 0:
            sim_infos = [
                c for c in evaluation_infos if c.status == _EvaluationStatus.TO_SIMULATE
            ]

            self.send_event(
                EverestStatusEvent(
                    batch=self._batch_id,
                    everest_event="START_OPTIMIZER_EVALUATION",
                )
            )

            logger.debug(f"Running forward model for batch {self._batch_id}")
            sim_objectives, sim_constraints = self._run_forward_model(
                sim_to_control_vector=control_values_to_simulate,
                sim_to_model_realization=[c.model_realization for c in sim_infos],
                sim_to_perturbation=[c.perturbation for c in sim_infos],
            )

            self.send_event(
                EverestBatchResultEvent(
                    batch=self._batch_id,
                    everest_event="FINISHED_OPTIMIZER_EVALUATION",
                    result_type="FunctionResult",
                )
            )

        else:
            sim_objectives = np.array([], dtype=np.float64)
            sim_constraints = np.array([], dtype=np.float64)

        # Assign simulated results to corresponding evaluation infos
        for ei in evaluation_infos:
            if ei.simulation_id is not None:
                ei.objectives = sim_objectives[ei.simulation_id]
                ei.constraints = (
                    sim_constraints[ei.simulation_id]
                    if sim_constraints is not None
                    else None
                )

        # At this point:
        # cached results are attached to cached evaluations
        # np.zeros are attached to inactive evaluations
        # simulated results are attached to simulated evaluations
        objectives = np.array(
            [cs.objectives for cs in evaluation_infos],
            dtype=np.float64,
        )

        constraints = (
            np.array(
                [cs.constraints for cs in evaluation_infos],
                dtype=np.float64,
            )
            if num_constraints > 0
            else None
        )

        # Also return for each control vector the simulation id and source batch.
        sim_ids = []
        source_batch_ids = []
        for ei in evaluation_infos:
            match ei.status:
                case _EvaluationStatus.TO_SIMULATE:
                    sim_ids.append(ei.simulation_id)
                    source_batch_ids.append(self._batch_id)
                case _EvaluationStatus.CACHED:
                    sim_id, batch_id = next(
                        (item["source_simulation_id"], item["source_batch_id"])
                        for item in cache_hits_dict
                        if item["target_evaluation_id"] == ei.flat_index
                    )
                    sim_ids.append(sim_id)
                    source_batch_ids.append(batch_id)
                case _EvaluationStatus.INACTIVE:
                    sim_ids.append(-1)
                    source_batch_ids.append(-1)

        evaluator_result = EvaluatorResult(
            objectives=objectives,
            constraints=constraints,
            batch_id=self._batch_id,
            evaluation_info={
                "sim_ids": np.array(sim_ids, dtype=np.int32),
                "source_batch_ids": np.array(source_batch_ids, dtype=np.int32),
            },
        )

        # increase the batch ID for the next evaluation:
        self._batch_id += 1

        return evaluator_result

    def _create_simulation_controls(
        self,
        control_values: NDArray[np.float64],
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
            for sim_id in range(control_values.shape[0])
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
        sim_to_model_realization: list[int],
    ) -> list[RunArg]:
        substitutions = self._substitutions
        self.active_realizations = [True] * len(sim_to_model_realization)
        for sim_id, model_realization in enumerate(sim_to_model_realization):
            substitutions[f"<GEO_ID_{sim_id}_{ensemble.iteration}>"] = str(
                int(model_realization)
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
        logger.debug("Simulation callback called")
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
                        logger.debug(f"Failed to remove {path}, {sys_info}")

                    shutil.rmtree(path_to_delete, onerror=onerror)

    def _gather_simulation_results(
        self, ensemble: Ensemble
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
        objective_names = self._everest_config.objective_names
        objectives = np.zeros((ensemble.ensemble_size, len(objective_names)))

        constraint_names = self._everest_config.constraint_names
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
            self._everest_config.simulation_dir is not None
            and os.path.exists(self._everest_config.simulation_dir)
            and any(os.listdir(self._everest_config.simulation_dir))
        )
