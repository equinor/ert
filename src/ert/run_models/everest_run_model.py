from __future__ import annotations

import copy
import dataclasses
import datetime
import importlib.metadata
import json
import logging
import os
import queue
import shutil
import traceback
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterator, MutableSequence
from enum import IntEnum, auto
from functools import cached_property
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Annotated, Any, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, PrivateAttr, TypeAdapter, ValidationError
from ropt.enums import ExitCode as RoptExitCode
from ropt.evaluator import EvaluatorContext, EvaluatorResult
from ropt.results import FunctionResults, Results
from ropt.transforms import OptModelTransforms
from ropt.workflow import BasicOptimizer
from typing_extensions import TypedDict

from ert.config import (
    EverestConstraintsConfig,
    EverestControl,
    EverestObjectivesConfig,
    GenDataConfig,
    HookRuntime,
    KnownQueueOptionsAdapter,
    QueueConfig,
    ResponseConfig,
    SummaryConfig,
    WorkflowJob,
)
from ert.config.ert_config import (
    create_and_hook_workflows,
    uppercase_subkeys_and_stringify_subvalues,
)
from ert.config.model_config import DEFAULT_ECLBASE_FORMAT
from ert.config.model_config import ModelConfig as ErtModelConfig
from ert.config.parsing import ConfigWarning
from ert.ensemble_evaluator import EndEvent, EvaluatorServerConfig
from ert.plugins import ErtRuntimePlugins
from ert.runpaths import Runpaths
from everest.config import (
    ControlConfig,
    EverestConfig,
    InputConstraintConfig,
    ModelConfig,
    OptimizationConfig,
)
from everest.config.forward_model_config import ForwardModelStepConfig, SummaryResults
from everest.everest_storage import EverestStorage
from everest.optimizer.everest2ropt import everest2ropt
from everest.optimizer.opt_model_transforms import (
    EverestOptModelTransforms,
    get_optimization_domain_transforms,
)
from everest.strings import EVEREST

from ..run_arg import RunArg, create_run_arguments
from ..storage import ExperimentState, ExperimentStatus
from ..storage.local_ensemble import EverestRealizationInfo
from ..substitutions import Substitutions
from .event import EverestBatchResultEvent, EverestStatusEvent
from .run_model import RunModel, RunModelConfig, StatusEvents

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


def _get_well_file(ever_config: EverestConfig) -> tuple[Path, str]:
    assert ever_config.output_dir is not None

    def _get_wells_from_controls(controls: list[ControlConfig]) -> list[str]:
        wells: list[str] = []
        for control in controls:
            if control.type != "well_control":
                continue
            for variable in control.variables:
                if variable.name not in wells:
                    wells.append(variable.name)
        return wells

    data_storage = (Path(ever_config.output_dir) / ".internal_data").resolve()
    return (
        data_storage / "wells.json",
        json.dumps(
            [{"name": name} for name in _get_wells_from_controls(ever_config.controls)]
            if ever_config.wells is None
            else [
                x.model_dump(exclude_none=True, exclude_unset=True)
                for x in ever_config.wells
            ]
        ),
    )


def _get_install_data_files(ever_config: EverestConfig) -> Iterator[tuple[Path, str]]:
    data_storage = (Path(ever_config.output_dir) / ".internal_data").resolve()
    for item in ever_config.install_data or []:
        if item.data is not None:
            target, data = item.inline_data_as_str()
            yield (data_storage / Path(target).name, data)


def _get_workflow_files(ever_config: EverestConfig) -> dict[str, tuple[Path, str]]:
    data_storage = (Path(ever_config.output_dir) / ".internal_data").resolve()
    return {
        trigger: (
            data_storage / f"{trigger}.workflow",
            "\n".join(getattr(ever_config.workflows, trigger, [])),
        )
        for trigger in ("pre_simulation", "post_simulation")
    }


def _get_internal_files(ever_config: EverestConfig) -> dict[Path, str]:
    return dict(
        [
            _get_well_file(ever_config),
            *(
                (workflow_file, jobs)
                for workflow_file, jobs in _get_workflow_files(ever_config).values()
                if jobs
            ),
            *_get_install_data_files(ever_config),
        ],
    )


def _get_workflow_jobs(ever_config: EverestConfig) -> dict[str, WorkflowJob]:
    workflow_jobs: dict[str, WorkflowJob] = {}
    for job in ever_config.install_workflow_jobs or []:
        workflow = job.to_ert_executable_workflow(ever_config.config_directory)
        if workflow.name in workflow_jobs:
            ConfigWarning.warn(
                f"Duplicate workflow job with name {job.name!r}, "
                f"overriding it with {job.executable!r}.",
                job.name,
            )
        workflow_jobs[workflow.name] = workflow
    return workflow_jobs


def _get_workflows(
    ever_config: EverestConfig,
) -> tuple[list[tuple[str, HookRuntime]], list[tuple[str, str]]]:
    trigger2res: dict[str, HookRuntime] = {
        "pre_simulation": HookRuntime.PRE_SIMULATION,
        "post_simulation": HookRuntime.POST_SIMULATION,
    }
    res_hooks: list[tuple[str, HookRuntime]] = []
    res_workflows: list[tuple[str, str]] = []
    for ever_trigger, (workflow_file, jobs) in _get_workflow_files(ever_config).items():
        if jobs:
            res_hooks.append((ever_trigger, trigger2res[ever_trigger]))
            res_workflows.append((str(workflow_file), ever_trigger))
    return res_hooks, res_workflows


EverestResponseTypes = (
    EverestObjectivesConfig | EverestConstraintsConfig | SummaryConfig | GenDataConfig
)

EverestResponseTypesAdapter = TypeAdapter(  # type: ignore
    Annotated[
        EverestResponseTypes,
        Field(discriminator="type"),
    ]
)


class EverestRunModelConfig(RunModelConfig):
    optimization_output_dir: str
    simulation_dir: str

    parameter_configuration: list[EverestControl]
    response_configuration: list[EverestResponseTypes]

    input_constraints: list[InputConstraintConfig]
    optimization: OptimizationConfig
    model: ModelConfig
    keep_run_path: bool
    experiment_name: str
    target_ensemble: str


class EverestRunModel(RunModel, EverestRunModelConfig):
    _exit_code: EverestExitCode | None = PrivateAttr(default=None)
    _experiment: Experiment | None = PrivateAttr(default=None)
    _eval_server_cfg: EvaluatorServerConfig | None = PrivateAttr(default=None)
    _batch_id: int = PrivateAttr(default=0)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

    @classmethod
    def create(
        cls,
        everest_config: EverestConfig,
        experiment_name: str = (
            f"EnOpt@{datetime.datetime.now().isoformat(timespec='seconds')}"
        ),
        target_ensemble: str = "batch",
        optimization_callback: OptimizerCallback | None = None,
        status_queue: queue.SimpleQueue[StatusEvents] | None = None,
        runtime_plugins: ErtRuntimePlugins | None = None,
    ) -> EverestRunModel:
        logger.info(
            "Using random seed: %d. To deterministically reproduce this experiment, "
            "add the above random seed to your configuration file.",
            everest_config.environment.random_seed,
        )

        if status_queue is None:
            status_queue = queue.SimpleQueue()

        runpath_file: Path = Path(
            os.path.join(everest_config.output_dir, ".res_runpath_list")
        )

        assert everest_config.config_file is not None
        config_file: Path = Path(everest_config.config_path)

        summary_fm = next(
            (
                fm
                for fm in everest_config.forward_model
                if fm.results is not None and isinstance(fm.results, SummaryResults)
            ),
            None,
        )

        parameter_configs = [
            control.to_ert_parameter_config() for control in everest_config.controls
        ]

        response_configs: list[ResponseConfig] = []

        response_configs.append(everest_config.create_ert_objectives_config())

        constraints_config = everest_config.create_ert_output_constraints_config()
        if constraints_config is not None:
            response_configs.append(constraints_config)

        gen_data_keys = [
            fm.results.file_name
            for fm in (everest_config.forward_model or [])
            if fm.results is not None and fm.results.type == "gen_data"
        ]

        if gen_data_keys:
            response_configs.append(
                GenDataConfig(
                    keys=gen_data_keys,
                    report_steps_list=[None] * len(gen_data_keys),
                    input_files=gen_data_keys,
                )
            )

        if summary_fm:
            assert isinstance(summary_fm.results, SummaryResults)
            eclbase = summary_fm.results.file_name
            response_configs.append(
                SummaryConfig(
                    keys=_extract_summary_keys(everest_config), input_files=[eclbase]
                )
            )
        else:
            eclbase = None

        runpath_config = ErtModelConfig(
            num_realizations=len(everest_config.model.realizations)
            if everest_config.model.realizations is not None
            else 1,
            runpath_format_string=str(
                Path(everest_config.simulation_dir)
                / "batch_<ITER>"
                / "realization_<REALIZATION_ID>"
                / "<SIM_DIR>"
            ),
            eclbase_format_string=eclbase
            if eclbase is not None
            else DEFAULT_ECLBASE_FORMAT,
        )

        simulator_config = everest_config.simulator
        queue_options_from_everconfig: dict[str, Any] = (
            {"name": "local"}
            if simulator_config.queue_system is None
            else (
                simulator_config.queue_system.model_dump(exclude_unset=True)
                | {
                    "name": simulator_config.queue_system.name,
                }
            )
        )

        if simulator_config.max_memory is not None:
            queue_options_from_everconfig["realization_memory"] = (
                simulator_config.max_memory or 0
            )

        # Number of cores reserved on queue nodes (NUM_CPU)
        if (num_fm_cpu := simulator_config.cores_per_node) is not None:
            if (
                simulator_config.queue_system is not None
                and "num_cpu" not in simulator_config.queue_system.model_fields_set
            ):
                queue_options_from_everconfig["num_cpu"] = num_fm_cpu
            else:
                warnings.warn(
                    "Ignoring cores_per_node as num_cpu was set",
                    UserWarning,
                    stacklevel=2,
                )

        # Only take into account site queue options
        # if and only if they exist and are of same type as user
        # specified queue system
        site_queue_options_to_apply = (
            runtime_plugins.queue_options.model_dump(exclude_unset=True)
            if (
                runtime_plugins is not None
                and runtime_plugins.queue_options is not None
                and runtime_plugins.queue_options.name
                == queue_options_from_everconfig["name"]
            )
            else {}
        )

        queue_options_dict = site_queue_options_to_apply | queue_options_from_everconfig

        queue_options = KnownQueueOptionsAdapter.validate_python(queue_options_dict)

        if queue_options.project_code is None:
            tags = {
                fm_name.lower()
                for fm_name in everest_config.forward_model_step_commands
                if fm_name.split(" ")[0].upper()
                in {"RMS", "FLOW", "ECLIPSE100", "ECLIPSE300"}
            }
            if tags:
                queue_options.project_code = "+".join(tags)

        queue_config = QueueConfig(
            max_submit=simulator_config.resubmit_limit + 1,
            queue_system=queue_options.name,
            queue_options=queue_options,
            stop_long_running=False,
            max_runtime=simulator_config.max_runtime,
        )

        substitutions = {
            "<RUNPATH_FILE>": str(runpath_file),
            "<RUNPATH>": runpath_config.runpath_format_string,
            "<ECL_BASE>": runpath_config.eclbase_format_string,
            "<ECLBASE>": runpath_config.eclbase_format_string,
            "<NUM_CPU>": str(queue_config.queue_options.num_cpu),
            "<CONFIG_PATH>": everest_config.config_directory,
            "<CONFIG_FILE>": Path(everest_config.config_file).stem,
        }

        for datafile, data in _get_internal_files(everest_config).items():
            datafile.parent.mkdir(exist_ok=True, parents=True)
            datafile.write_text(data, encoding="utf-8")

        workflow_jobs = _get_workflow_jobs(everest_config)
        hooks, workflows = _get_workflows(everest_config)
        _, hooked_workflows = create_and_hook_workflows(
            hooks, workflows, workflow_jobs, substitutions
        )

        install_job_fm_steps = {
            job.name: job.to_ert_forward_model_step(
                config_directory=everest_config.config_directory
            )
            for job in everest_config.install_jobs
        }

        site_installed_fm_steps = (
            runtime_plugins.installed_forward_model_steps
            if runtime_plugins is not None
            else {}
        )
        installed_fm_steps = dict(site_installed_fm_steps) | install_job_fm_steps

        install_data_fm_steps = [
            install_data.to_ert_forward_model_step(
                config_directory=everest_config.config_directory,
                output_directory=everest_config.output_dir,
                model_realizations=everest_config.model.realizations,
                installed_fm_steps=installed_fm_steps,
            )
            for install_data in everest_config.install_data
        ]

        well_path, _ = _get_well_file(everest_config)
        copy_wellfile = copy.deepcopy(installed_fm_steps.get("copy_file"))
        assert copy_wellfile is not None
        copy_wellfile.arglist = [str(well_path), str(well_path.name)]

        # map templating to template_render job
        template_fm_steps = [
            tmpl_request.to_ert_forward_model_step(
                control_names=[control.name for control in everest_config.controls],
                installed_fm_steps=installed_fm_steps,
                well_path=str(well_path),
            )
            for tmpl_request in everest_config.install_templates
        ]

        user_fm_steps = [
            fm_spec.to_ert_forward_model_step(installed_fm_steps)
            for fm_spec in everest_config.forward_model
        ]

        forward_model_steps = [
            *install_data_fm_steps,
            copy_wellfile,
            *template_fm_steps,
            *user_fm_steps,
        ]

        env_pr_fm_step = uppercase_subkeys_and_stringify_subvalues(
            {
                k: dict(v)
                for k, v in (
                    runtime_plugins.env_pr_fm_step
                    if runtime_plugins is not None
                    else {}
                ).items()
            }
        )

        env_vars = {}
        plugin_env_vars = (
            runtime_plugins.environment_variables if runtime_plugins else {}
        )
        substituter = Substitutions(substitutions)

        if runtime_plugins is not None:
            for key, val in plugin_env_vars.items():
                env_vars[key] = substituter.substitute(val)

        delete_run_path: bool = (
            everest_config.simulator is not None
            and everest_config.simulator.delete_run_path
        )

        return cls(
            experiment_name=experiment_name,
            target_ensemble=target_ensemble,
            controls=everest_config.controls,
            simulation_dir=everest_config.simulation_dir,
            keep_run_path=not delete_run_path,
            objective_names=everest_config.objective_names,
            objective_functions=everest_config.objective_functions,
            input_constraints=everest_config.input_constraints,
            optimization=everest_config.optimization,
            model=everest_config.model,
            optimization_output_dir=everest_config.optimization_output_dir,
            log_path=everest_config.log_dir,
            random_seed=everest_config.environment.random_seed,
            runpath_file=runpath_file,
            # Mutated throughout execution of Everest
            # (Not totally in conformity with ERT runmodel logic)
            active_realizations=[],
            parameter_configuration=parameter_configs,
            response_configuration=response_configs,
            user_config_file=config_file,
            env_vars=env_vars,
            env_pr_fm_step=env_pr_fm_step,
            runpath_config=runpath_config,
            forward_model_steps=forward_model_steps,
            substitutions=substitutions,
            hooked_workflows=hooked_workflows,
            storage_path=str(everest_config.storage_dir),
            queue_config=queue_config,
            status_queue=status_queue,
            optimization_callback=optimization_callback,
        )

    @property
    def _everest_control_configs(self) -> list[EverestControl]:
        return [
            c for c in self.parameter_configuration if c.type == "everest_parameters"
        ]

    @cached_property
    def _transforms(self) -> EverestOptModelTransforms:
        return get_optimization_domain_transforms(
            self._everest_control_configs,
            self.objectives_config,
            self.input_constraints,
            self.output_constraints_config,
            self.model,
            self.optimization.auto_scale,
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

    def cancel(self) -> None:
        if self._experiment is not None:
            self._experiment.status = ExperimentStatus(
                message="Optimization aborted",
                status=ExperimentState.stopped,
            )
        super().cancel()

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
                        else "Experiment completed"
                    ),
                )
            )

    def _handle_optimizer_results(self, results: tuple[Results, ...]) -> None:
        assert self._experiment is not None

        batch_dataframes = EverestStorage.unpack_ropt_results(results)

        for batch_id, batch_dict in batch_dataframes.items():
            target_ensemble = self._experiment.get_ensemble_by_name(f"batch_{batch_id}")
            target_ensemble.save_batch_dataframes(dataframes=batch_dict)
            target_ensemble.update_improvement_flag(is_improvement=False)

        for r in results:
            batches = (
                self._experiment.ensembles_with_function_results
                if isinstance(r, FunctionResults)
                else self._experiment.ensembles_with_gradient_results
            )
            ens = next((ens for ens in batches if ens.iteration == r.batch_id), None)
            if ens is None:
                continue

            results_dict: dict[str, Any] | None = None
            if isinstance(r, FunctionResults):
                results_dict = {}
                if ens.realization_controls is not None:
                    results_dict |= {
                        "controls": ens.realization_controls.drop(
                            "realization", "simulation_id"
                        ).to_dicts()[0],
                    }

                if ens.realization_objectives is not None:
                    results_dict |= {
                        "realization_objectives": ens.realization_objectives.drop(
                            "batch_id"
                        ).to_dicts()
                    }

                if ens.batch_objectives is not None:
                    results_dict |= {
                        "objectives": ens.batch_objectives.drop(
                            "batch_id", "total_objective_value"
                        ).to_dicts()[0],
                        "total_objective_value": ens.batch_objectives[
                            "total_objective_value"
                        ].item(),
                    }

                if ens.realization_constraints is not None:
                    results_dict |= {
                        "realization_constraints": ens.realization_constraints.drop(
                            "batch_id"
                        ).to_dicts()
                    }
            else:
                results_dict = {}
                objective_gradient = (
                    ens.batch_objective_gradient.drop("batch_id")
                    .sort("control_name")
                    .to_dicts()
                    if ens.batch_objective_gradient is not None
                    else None
                )

                if objective_gradient is not None:
                    results_dict |= {"objective_gradient_values": objective_gradient}

                perturbation_objectives = (
                    (
                        ens.perturbation_objectives.drop("batch_id")
                        .sort("realization", "perturbation")
                        .to_dicts()
                    )
                    if ens.perturbation_objectives is not None
                    else None
                )

                if perturbation_objectives is not None:
                    results_dict |= {"perturbation_objectives": perturbation_objectives}

                constraint_gradient_dicts = (
                    ens.batch_constraint_gradient.drop("batch_id")
                    .sort("control_name")
                    .to_dicts()
                    if ens.batch_constraint_gradient is not None
                    else None
                )

                if constraint_gradient_dicts is not None:
                    results_dict |= {"constraint_gradient": constraint_gradient_dicts}

                perturbation_gradient_dicts = (
                    ens.perturbation_constraints.drop("batch_id")
                    .sort("realization", "perturbation")
                    .to_dicts()
                    if ens.perturbation_constraints is not None
                    else None
                )

                if perturbation_gradient_dicts is not None:
                    results_dict |= {
                        "perturbation_constraints": perturbation_gradient_dicts
                    }

            self.send_event(
                EverestBatchResultEvent(
                    batch=r.batch_id,
                    everest_event="OPTIMIZATION_RESULT",
                    result_type="FunctionResult"
                    if isinstance(r, FunctionResults)
                    else "GradientResult",
                    results=results_dict,
                )
            )

    @property
    def output_constraints_config(self) -> EverestConstraintsConfig | None:
        constraints_config = next(
            (c for c in self.response_configuration if c.type == "everest_constraints"),
            None,
        )

        if constraints_config is None:
            return None

        assert isinstance(constraints_config, EverestConstraintsConfig)
        return constraints_config

    @property
    def objectives_config(self) -> EverestObjectivesConfig:
        obj_config = next(
            c for c in self.response_configuration if c.type == "everest_objectives"
        )
        # There will and must always be one objectives config for an
        # Everest optimization.
        assert isinstance(obj_config, EverestObjectivesConfig)
        return obj_config

    def _update_ensemble_improvement_flags(self) -> None:
        assert self._experiment is not None

        # This a somewhat arbitrary threshold, this should be a user choice
        # during visualization:
        CONSTRAINT_TOL = 1e-6

        max_total_objective = -np.inf
        for ensemble in self._experiment.ensembles_with_function_results:
            assert ensemble.batch_objectives is not None
            total_objective = ensemble.batch_objectives["total_objective_value"].item()
            bound_constraint_violation = (
                0.0
                if ensemble.batch_bound_constraint_violations is None
                else (
                    ensemble.batch_bound_constraint_violations.drop("batch_id")
                    .to_numpy()
                    .min()
                    .item()
                )
            )
            input_constraint_violation = (
                0.0
                if ensemble.batch_input_constraint_violations is None
                else (
                    ensemble.batch_input_constraint_violations.drop("batch_id")
                    .to_numpy()
                    .min()
                    .item()
                )
            )
            output_constraint_violation = (
                0.0
                if ensemble.batch_output_constraint_violations is None
                else (
                    ensemble.batch_output_constraint_violations.drop("batch_id")
                    .to_numpy()
                    .min()
                    .item()
                )
            )
            if (
                max(
                    bound_constraint_violation,
                    input_constraint_violation,
                    output_constraint_violation,
                )
                < CONSTRAINT_TOL
                and total_objective > max_total_objective
            ):
                ensemble.update_improvement_flag(is_improvement=True)
                max_total_objective = total_objective

    def run_experiment(
        self,
        evaluator_server_config: EvaluatorServerConfig,
        rerun_failed_realizations: bool = False,
    ) -> None:
        self.log_at_startup()
        self._eval_server_cfg = evaluator_server_config

        self._experiment = self._experiment or self._storage.create_experiment(
            name=self.experiment_name, experiment_config=self.model_dump(mode="json")
        )

        self._experiment.status = ExperimentStatus(
            message="Experiment started", status=ExperimentState.running
        )

        # Initialize the ropt optimizer:
        optimizer, initial_guesses = self._create_optimizer()

        # ROPT expects this folder to exist wrt stdout/stderr redirect files
        Path(self.optimization_output_dir).mkdir(exist_ok=True)
        optimizer.set_results_callback(self._handle_optimizer_results)

        # Run the optimization:
        optimizer_exit_code = optimizer.run(initial_guesses)

        # Store some final results.
        self._update_ensemble_improvement_flags()
        if (
            optimizer_exit_code is not RoptExitCode.UNKNOWN
            and optimizer_exit_code is not RoptExitCode.TOO_FEW_REALIZATIONS
            and optimizer_exit_code is not RoptExitCode.USER_ABORT
        ):
            self._experiment.export_everest_opt_results_to_csv()

        experiment_status = None
        if self._exit_code is None:
            match optimizer_exit_code:
                case RoptExitCode.MAX_FUNCTIONS_REACHED:
                    self._exit_code = EverestExitCode.MAX_FUNCTIONS_REACHED
                    experiment_status = ExperimentStatus(
                        message="Maximum number of function evaluations reached",
                        status=ExperimentState.completed,
                    )
                case RoptExitCode.MAX_BATCHES_REACHED:
                    self._exit_code = EverestExitCode.MAX_BATCH_NUM_REACHED
                    experiment_status = ExperimentStatus(
                        message="Maximum number of batches reached",
                        status=ExperimentState.completed,
                    )
                case RoptExitCode.USER_ABORT:
                    self._exit_code = EverestExitCode.USER_ABORT
                    experiment_status = ExperimentStatus(
                        message="Optimization aborted", status=ExperimentState.stopped
                    )
                case RoptExitCode.TOO_FEW_REALIZATIONS:
                    self._exit_code = (
                        EverestExitCode.TOO_FEW_REALIZATIONS
                        if self.get_number_of_successful_realizations() > 0
                        else EverestExitCode.ALL_REALIZATIONS_FAILED
                    )
                    experiment_status = ExperimentStatus(
                        message="Too few realizations are evaluated successfully"
                        if self.get_number_of_successful_realizations() > 0
                        else "All realizations failed",
                        status=ExperimentState.failed,
                    )
                case _:
                    self._exit_code = EverestExitCode.COMPLETED
                    experiment_status = ExperimentStatus(
                        message="Experiment completed", status=ExperimentState.completed
                    )

        if experiment_status is not None:
            self._experiment.status = experiment_status

        logger.debug(
            f"Everest experiment finished with exit code {self._exit_code.name}"
        )

    def _create_optimizer(self) -> tuple[BasicOptimizer, list[float]]:
        enopt_config, initial_guesses = everest2ropt(
            self.parameter_configuration,
            self.objectives_config,
            self.input_constraints,
            self.output_constraints_config,
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
            name=f"{self.target_ensemble}_{self._batch_id}",
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

        iens = sim_to_control_vector.shape[0]
        offset = 0
        for control_config in self._everest_control_configs:
            n_param_keys = len(control_config.parameter_keys)
            name = control_config.name
            parameters = sim_to_control_vector[:, offset : offset + n_param_keys]
            ensemble.save_parameters_numpy(
                parameters.reshape(-1, n_param_keys),
                name,
                np.arange(iens),
            )
            offset += n_param_keys

        # Evaluate the batch:
        run_args = self._get_run_args(
            ensemble, sim_to_model_realization, sim_to_perturbation
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

            # Calculate auto-scales if necessary. Skip this if there are any
            # objectives or constraints where all realizations failed. In that
            # case the auto-scale calculations will fail, and the optimization
            # will terminate afterwards in any case.
            if not np.any(np.all(np.isnan(objectives), axis=0)):
                self._calculate_objective_auto_scales(
                    objectives, realization_indices, perturbation_indices
                )
            if constraints is not None and not np.any(
                np.all(np.isnan(constraints), axis=0)
            ):
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
                (num_all_simulations, len(self.objectives_config.keys)),
                dtype=np.float64,
            )
            constraints = (
                np.zeros(
                    (num_all_simulations, len(self.output_constraints_config.keys)),
                    dtype=np.float64,
                )
                if self.output_constraints_config
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
        sim_to_perturbation: list[int],
    ) -> list[RunArg]:
        substitutions = self.substitutions
        self.active_realizations = [True] * len(sim_to_model_realization)

        # Function evalutions do not have a number/id yet, so we index
        # them from zero in each model realization:
        eval_idx: defaultdict[int, int] = defaultdict(lambda: 0)
        for sim_id, (model_realization, perturbation) in enumerate(
            zip(sim_to_model_realization, sim_to_perturbation, strict=True)
        ):
            substitutions[f"<REALIZATION_ID_{sim_id}_{ensemble.iteration}>"] = str(
                int(model_realization)
            )
            if perturbation >= 0:
                substitutions[f"<SIM_DIR_{sim_id}_{ensemble.iteration}>"] = (
                    f"perturbation_{perturbation}"
                )
            else:
                substitutions[f"<SIM_DIR_{sim_id}_{ensemble.iteration}>"] = (
                    f"evaluation_{eval_idx[model_realization]}"
                )
                eval_idx[model_realization] += 1

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
        objective_names = self.objectives_config.keys
        objectives = np.zeros((ensemble.ensemble_size, len(objective_names)))

        constraint_names = (
            self.output_constraints_config.keys
            if self.output_constraints_config is not None
            else []
        )
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
        return os.path.exists(self.simulation_dir) and any(
            os.listdir(self.simulation_dir)
        )


def _extract_summary_keys(ever_config: EverestConfig) -> list[str]:
    DEFAULT_DATA_SUMMARY_KEYS = ["YEAR", "YEARS", "TCPU", "TCPUDAY", "MONTH", "DAY"]

    DEFAULT_FIELD_SUMMARY_KEYS = [
        "FOPR",
        "FOPT",
        "FOIR",
        "FOIT",
        "FWPR",
        "FWPT",
        "FWIR",
        "FWIT",
        "FGPR",
        "FGPT",
        "FGIR",
        "FGIT",
        "FVPR",
        "FVPT",
        "FVIR",
        "FVIT",
        "FWCT",
        "FGOR",
        "FOIP",
        "FOIPL",
        "FOIPG",
        "FWIP",
        "FGIP",
        "FGIPL",
        "FGIPG",
        "FPR",
        "FAQR",
        "FAQRG",
        "FAQT",
        "FAQTG",
        "FWGR",
    ]

    DEFAULT_WELL_SUMMARY_KEYS = [
        "WOPR",
        "WOPT",
        "WOIR",
        "WOIT",
        "WWPR",
        "WWPT",
        "WWIR",
        "WWIT",
        "WGPR",
        "WGPT",
        "WGIR",
        "WGIT",
        "WVPR",
        "WVPT",
        "WVIR",
        "WVIT",
        "WWCT",
        "WGOR",
        "WWGR",
        "WBHP",
        "WTHP",
        "WPI",
    ]

    DEFAULT_WELL_TARGET_SUMMARY_KEYS = [
        well_key + "T"
        for well_key in DEFAULT_WELL_SUMMARY_KEYS
        if well_key.endswith("R") and well_key != "WGOR"
    ]

    summary_fms: list[ForwardModelStepConfig] = [
        fm
        for fm in ever_config.forward_model
        if fm.results is not None and fm.results.type == "summary"
    ]

    if not summary_fms:
        return []

    smry_results = summary_fms[0].results
    assert isinstance(smry_results, SummaryResults)

    requested_keys: list[str] = ["*"] if smry_results.keys == "*" else smry_results.keys

    well_keys = [
        f"{sum_key}:*"
        for sum_key in DEFAULT_WELL_SUMMARY_KEYS + DEFAULT_WELL_TARGET_SUMMARY_KEYS
    ]
    deprecated_user_specified_keys = (
        [] if ever_config.export is None else ever_config.export.keywords
    )

    return list(
        set(
            requested_keys
            + DEFAULT_DATA_SUMMARY_KEYS
            + DEFAULT_FIELD_SUMMARY_KEYS
            + well_keys
            + deprecated_user_specified_keys
        )
    )
