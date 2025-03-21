import logging
import os
from argparse import ArgumentParser
from copy import copy
from io import StringIO
from itertools import chain
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Optional,
    Protocol,
    Self,
    no_type_check,
)

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import ErrorDetails
from ruamel.yaml import YAML, YAMLError

from ert.config import ErtConfig
from ert.config.parsing.queue_system import QueueSystem
from everest.config.control_variable_config import ControlVariableGuessListConfig
from everest.config.install_template_config import InstallTemplateConfig
from everest.config.server_config import ServerConfig
from everest.config.validation_utils import (
    InstallDataContext,
    check_for_duplicate_names,
    check_path_exists,
    check_writeable_path,
    unique_items,
    validate_forward_model_configs,
)
from everest.jobs import script_names
from everest.util.forward_models import (
    validate_forward_model_step_arguments,
)

from ..config_file_loader import yaml_file_to_substituted_config_dict
from ..strings import (
    DEFAULT_OUTPUT_DIR,
    EVEREST,
    OPTIMIZATION_LOG_DIR,
    OPTIMIZATION_OUTPUT_DIR,
    STORAGE_DIR,
)
from .control_config import ControlConfig
from .environment_config import EnvironmentConfig
from .export_config import ExportConfig
from .input_constraint_config import InputConstraintConfig
from .install_data_config import InstallDataConfig
from .install_job_config import InstallJobConfig
from .model_config import ModelConfig
from .objective_function_config import ObjectiveFunctionConfig
from .optimization_config import OptimizationConfig
from .output_constraint_config import OutputConstraintConfig
from .simulator_config import SimulatorConfig, simulator_example
from .well_config import WellConfig
from .workflow_config import WorkflowConfig


def _dummy_ert_config() -> ErtConfig:
    site_config = ErtConfig.read_site_config()
    dummy_config = {"NUM_REALIZATIONS": 1, "ENSPATH": "."}
    dummy_config.update(site_config)  # type: ignore
    return ErtConfig.with_plugins().from_dict(config_dict=dummy_config)


def get_system_installed_jobs() -> list[str]:
    """Returns list of all system installed job names"""
    return list(_dummy_ert_config().installed_forward_model_steps.keys())


class EverestValidationError(ValueError):
    def __init__(self) -> None:
        super().__init__()
        self.errors: list[tuple[ErrorDetails, tuple[int, int] | None]] = []

    @property
    def error(self) -> list[tuple[ErrorDetails, tuple[int, int] | None]]:
        return self.errors

    def __str__(self) -> str:
        return f"{self.errors!s}"


def _error_loc(error_dict: ErrorDetails) -> str:
    return " -> ".join(
        str(e) for e in error_dict["loc"] if e is not None and e != "__root__"
    )


def format_errors(validation_error: EverestValidationError) -> str:
    msg = f"Found {len(validation_error.errors)} validation error{'s' if len(validation_error.errors) > 1 else ''}:\n\n"
    error_map = {}
    for error, pos in validation_error.errors:
        if pos:
            row, col = pos
            key = f"line: {row}, column: {col}. {_error_loc(error)}"
        else:
            key = f"{_error_loc(error)}"
        if key not in error_map:
            error_map[key] = [key]
        error_map[key].append(f"    * {error['msg']} (type={error['type']})")
    return msg + "\n".join(list(chain.from_iterable(error_map.values())))


class HasName(Protocol):
    name: str


class EverestConfig(BaseModel):
    controls: Annotated[list[ControlConfig], AfterValidator(unique_items)] = Field(
        description="""Defines a list of controls.
         Controls should have unique names each control defines
            a group of control variables
        """,
        min_length=1,
    )
    objective_functions: list[ObjectiveFunctionConfig] = Field(
        description="List of objective function specifications", min_length=1
    )
    optimization: OptimizationConfig = Field(
        default_factory=OptimizationConfig,
        description="Optimizer options",
    )
    model: ModelConfig = Field(
        description="Configuration of the Everest model",
    )
    environment: EnvironmentConfig = Field(
        default_factory=EnvironmentConfig,
        description="The environment of Everest, specifies which folders are used "
        "for simulation and output, as well as the level of detail in Everest-logs",
    )
    wells: list[WellConfig] = Field(
        default_factory=list,
        description="A list of well configurations, all with unique names.",
    )
    definitions: dict[str, Any] = Field(
        default_factory=dict[str, Any],
        description="""Section for specifying variables.

Used to specify variables that will be replaced in the file when encountered.

| scratch: /scratch/ert/
| num_reals: 10
| min_success: 13
| fixed_wells: [Prod1, Inj3]

Some keywords are pre-defined by Everest,

| realization: <GEO_ID>
| configpath: <CONFIG_PATH>
| runpath_file: <RUNPATH_FILE>
| eclbase: <ECLBASE>

and environment variables are exposed in the form 'os.NAME', for example:

| os.USER: $USER
| os.HOSTNAME: $HOSTNAME
| ...
    """,
    )
    input_constraints: list[InputConstraintConfig] | None = Field(
        default=None, description="List of input constraints"
    )
    output_constraints: list[OutputConstraintConfig] | None = Field(
        default=None, description="A list of output constraints with unique names."
    )
    install_jobs: list[InstallJobConfig] | None = Field(
        default=None, description="A list of jobs to install"
    )
    install_workflow_jobs: list[InstallJobConfig] | None = Field(
        default=None, description="A list of workflow jobs to install"
    )
    install_data: list[InstallDataConfig] | None = Field(
        default=None,
        description="""A list of install data elements from the install_data config
        section. Each item marks what folders or paths need to be copied or linked
        in order for the evaluation jobs to run.""",
    )
    install_templates: list[InstallTemplateConfig] | None = Field(
        default=None,
        description="""Allow the user to define the workflow establishing the model
        chain for the purpose of sensitivity analysis, enabling the relationship
        between sensitivity input variables and quantities of interests to be
        evaluated.
""",
    )
    server: ServerConfig | None = Field(
        default_factory=ServerConfig,
        description="""Defines Everest server settings, i.e., which queue system,
            queue name and queue options are used for the everest server.
            The main reason for changing this section is situations where everest
            times out because it can not add the server to the queue.
            This makes it possible to reduce the resource requirements as they tend to
            be low compared with the forward model.

            Queue system and queue name defaults to the same as simulator, and the
            server should not need to be configured by most users.
            This is also true for the --include-host and --exclude-host options
            that are used by the SLURM driver.

            Note that changing values in this section has no impact on the resource
            requirements of the forward models.
""",
    )
    simulator: SimulatorConfig | None = Field(
        default_factory=SimulatorConfig,
        description="Simulation settings",
        examples=[simulator_example],
    )
    forward_model: list[str] | None = Field(
        default=None, description="List of jobs to run"
    )
    workflows: WorkflowConfig | None = Field(
        default=None, description="Workflows to run during optimization"
    )
    export: ExportConfig | None = Field(
        default=None,
        description="Settings to control the exports of a optimization run by everest.",
    )
    config_path: Path = Field()
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def validate_queue_system(self) -> Self:  # pylint: disable=E0213
        assert self.server is not None
        assert self.simulator is not None
        if self.server.queue_system is None:
            self.server.queue_system = copy(self.simulator.queue_system)
        if (
            str(self.simulator.queue_system.name).lower() == "local"  # type: ignore
            and str(self.server.queue_system.name).lower()  # type: ignore
            != str(self.simulator.queue_system.name).lower()  # type: ignore
        ):
            raise ValueError(
                f"The simulator is using local as queue system "
                f"while the everest server is using {self.server.queue_system.name}. "  # type: ignore
                f"If the simulator is using local, so must the everest server."
            )
        self.server.queue_system.max_running = 1  # type: ignore
        return self

    @model_validator(mode="after")
    def validate_forward_model_job_name_installed(self) -> Self:  # pylint: disable=E0213
        install_jobs = self.install_jobs
        forward_model_jobs = self.forward_model
        if install_jobs is None:
            install_jobs = []
        if not forward_model_jobs:
            return self
        installed_jobs_name = [job.name for job in install_jobs]
        installed_jobs_name += list(script_names)  # default jobs
        installed_jobs_name += get_system_installed_jobs()  # system jobs

        errors = []
        for fm_job in forward_model_jobs:
            job_name = fm_job.split()[0]
            if job_name not in installed_jobs_name:
                errors.append(f"unknown job {job_name}")

        if len(errors) > 0:  # Note: python3.11 ExceptionGroup will solve this nicely
            raise ValueError(errors)
        return self

    @model_validator(mode="after")
    def validate_workflow_name_installed(self) -> Self:  # pylint: disable=E0213
        workflows = self.workflows
        if workflows is None:
            return self

        install_workflow_jobs = self.install_workflow_jobs
        if install_workflow_jobs is None:
            install_workflow_jobs = []
        installed_jobs_name = [job.name for job in install_workflow_jobs]

        errors = []
        workflows_dict = workflows.model_dump()
        for trigger in ("pre_simulation", "post_simulation"):
            trigger_jobs = workflows_dict.get(trigger)
            if trigger_jobs is None:
                continue
            for workflow_job in trigger_jobs:
                job_name = workflow_job.split()[0]
                if job_name not in installed_jobs_name:
                    errors.append(f"unknown workflow job {job_name}")

        if len(errors) > 0:  # Note: python3.11 ExceptionGroup will solve this nicely
            raise ValueError(errors)
        return self

    @field_validator("install_templates")
    @classmethod
    def validate_install_templates_unique_output_files(
        cls, install_templates: list[InstallTemplateConfig] | None
    ) -> list[InstallTemplateConfig] | None:
        if install_templates is None:
            return None
        check_for_duplicate_names(
            [t.output_file for t in install_templates],
            "install_templates",
            "output_file",
        )
        return install_templates

    @model_validator(mode="after")
    def validate_install_templates_are_existing_files(self) -> Self:
        install_templates = self.install_templates

        if not install_templates:
            return self

        config_path = self.config_path
        model = self.model

        if not config_path or not model:
            return self

        template_paths = [t.template for t in install_templates]
        errors = []
        for template_path in template_paths:
            try:
                check_path_exists(
                    path_source=template_path,
                    config_path=config_path,
                    realizations=model.realizations,
                )
            except ValueError as e:
                errors.append(str(e))

        if len(errors) > 0:
            raise ValueError(errors)

        return self

    @model_validator(mode="after")
    def validate_cvar_nreals_interval(self) -> Self:  # pylint: disable=E0213
        optimization = self.optimization
        if not optimization:
            return self

        cvar = optimization.cvar

        if not cvar:
            return self

        nreals = cvar.number_of_realizations
        model = self.model

        if (
            cvar
            and nreals is not None
            and model is not None
            and not (0 < nreals < len(model.realizations))
        ):
            raise ValueError(
                f"number_of_realizations: (got {nreals}) was"
                f" expected to be between 0 and number of realizations specified "
                f"in model config: {len(model.realizations)}"
            )

        return self

    @model_validator(mode="after")
    def validate_install_data_source_exists(self) -> Self:
        install_data = self.install_data or []
        if not install_data:
            return self
        config_path = self.config_path
        model = self.model
        if model and config_path:
            for install_data_cfg in install_data:
                check_path_exists(
                    install_data_cfg.source, config_path, model.realizations
                )

        return self

    @model_validator(mode="after")
    def validate_model_data_file_exists(self) -> Self:  # pylint: disable=E0213
        model = self.model
        if not model:
            return self
        config_path = self.config_path

        if model.data_file is not None:
            check_path_exists(model.data_file, config_path, model.realizations)

        return self

    @model_validator(mode="after")
    def validate_maintained_forward_models(self) -> Self:
        install_data = self.install_data

        with InstallDataContext(install_data, self.config_path) as context:  # type: ignore
            for realization in self.model.realizations:
                context.add_links_for_realization(realization)
            validate_forward_model_configs(self.forward_model, self.install_jobs)
        return self

    @model_validator(mode="after")
    def validate_maintained_forward_model_step_arguments(self) -> Self:
        if not self.forward_model:
            return self
        validate_forward_model_step_arguments(self.forward_model)

        return self

    @model_validator(mode="after")
    # pylint: disable=E0213
    def validate_input_constraints_weight_definition(self) -> Self:
        input_constraints = self.input_constraints
        if not input_constraints:
            return self

        controls = self.controls
        if controls is None:
            return self
        control_names = self.formatted_control_names
        control_names_deprecated = self.formatted_control_names_dotdash
        errors = []

        for input_const in input_constraints:
            for key in input_const.weights:
                if key in control_names_deprecated and key not in control_names:
                    logging.getLogger(EVEREST).warning(
                        f"Deprecated reference to control: {key} in input constraint."
                    )
                    print(
                        f"Deprecated input control name: {key} "
                        f"reference in input constraint. This format is deprecated, "
                        f"please use only '.' as delimiters: {key.replace('-', '.')}"
                    )
                elif key not in control_names and key not in control_names_deprecated:
                    errors.append(
                        f"Input control weight name {key} "
                        f"does not match any instance of "
                        f"control_name.variable_name.variable_index"
                    )

        if len(errors) > 0:  # Note: python3.11 ExceptionGroup will solve this nicely
            raise ValueError(errors)

        return self

    @model_validator(mode="after")
    def validate_variable_name_match_well_name(self) -> Self:  # pylint: disable=E0213
        controls = self.controls
        wells = self.wells
        if controls is None or wells is None:
            return self
        well_names = [w.name for w in wells]
        if not well_names:
            return self
        for c in controls:
            if c.type == "generic_control":
                continue
            for v in c.variables:
                if v.name not in well_names:
                    raise ValueError("Variable name does not match any well name")

        return self

    @model_validator(mode="after")
    def validate_that_environment_sim_folder_is_writeable(self) -> Self:
        environment = self.environment
        config_path = self.config_path
        if environment is None or config_path is None:
            return self

        check_writeable_path(environment.simulation_folder, Path(config_path))
        return self

    # pylint: disable=E0213
    @field_validator("wells")
    @no_type_check
    @classmethod
    def validate_unique_well_names(cls, wells: list[WellConfig]):
        check_for_duplicate_names([w.name for w in wells], "well", "name")
        return wells

    # pylint: disable=E0213
    @field_validator("output_constraints")
    @no_type_check
    @classmethod
    def validate_unique_output_constraint_names(
        cls, output_constraints: list[OutputConstraintConfig]
    ):
        check_for_duplicate_names(
            [c.name for c in output_constraints], "output constraint", "name"
        )
        return output_constraints

    @field_validator("objective_functions")
    @no_type_check
    @classmethod
    def validate_objective_function_weights_for_all_or_none(cls, functions):
        objective_names = [function.name for function in functions]
        weights = [
            function.weight for function in functions if function.weight is not None
        ]
        if weights and len(weights) != len(objective_names):
            raise ValueError(
                "Weight should be given either for all of the objectives"
                " or for none of them"
            )
        return functions

    @field_validator("config_path")
    @no_type_check
    @classmethod
    def validate_config_path_exists(cls, config_path):
        expanded_path = os.path.realpath(config_path)
        if not os.path.exists(expanded_path):
            raise ValueError(f"no such file or directory {expanded_path}")
        return config_path

    def copy(self) -> "EverestConfig":  # type: ignore
        return EverestConfig.model_validate(self.model_dump(exclude_none=True))

    @property
    def logging_level(self) -> int:
        level = self.environment.log_level

        levels = {
            "debug": logging.DEBUG,  # 10
            "info": logging.INFO,  # 20
            "warning": logging.WARNING,  # 30
            "error": logging.ERROR,  # 40
            "critical": logging.CRITICAL,  # 50
        }
        return levels.get(level.lower(), logging.INFO)

    @property
    def config_directory(self) -> str:
        return str(self.config_path.parent)

    @property
    def config_file(self) -> str:
        return self.config_path.name

    @property
    def output_dir(self) -> str:
        path = self.environment.output_folder

        if path is None:
            path = DEFAULT_OUTPUT_DIR

        if os.path.isabs(path):
            return path

        cfgdir = self.config_directory

        if cfgdir is None:
            return path

        return os.path.join(cfgdir, path)

    @property
    def simulation_dir(self) -> str | None:
        path = self.environment.simulation_folder

        if os.path.isabs(path):
            return path

        cfgdir = self.output_dir
        if cfgdir is None:
            return path

        return os.path.join(cfgdir, path)

    def _get_output_subdirectory(self, subdirname: str) -> str:
        return os.path.join(os.path.abspath(self.output_dir), subdirname)

    @property
    def optimization_output_dir(self) -> str:
        """Return the path to folder with the optimization output"""
        return self._get_output_subdirectory(OPTIMIZATION_OUTPUT_DIR)

    @property
    def storage_dir(self) -> str:
        return self._get_output_subdirectory(STORAGE_DIR)

    @property
    def log_dir(self) -> str:
        return self._get_output_subdirectory(OPTIMIZATION_LOG_DIR)

    @property
    def control_names(self) -> list[str]:
        controls = self.controls or []
        return [control.name for control in controls]

    @property
    def formatted_control_names(self) -> list[str]:
        names = []
        for control in self.controls:
            for variable in control.variables:
                if isinstance(variable, ControlVariableGuessListConfig):
                    for index in range(1, len(variable.initial_guess) + 1):
                        names.append(f"{control.name}.{variable.name}.{index}")
                elif variable.index is not None:
                    names.append(f"{control.name}.{variable.name}.{variable.index}")
                else:
                    names.append(f"{control.name}.{variable.name}")
        return names

    @property
    def formatted_control_names_dotdash(self) -> list[str]:
        # Note: Should be removed as the .- way of referencing controls
        # from input constraints is removed
        names = []
        for control in self.controls:
            for variable in control.variables:
                if isinstance(variable, ControlVariableGuessListConfig):
                    for index in range(1, len(variable.initial_guess) + 1):
                        names.append(f"{control.name}.{variable.name}-{index}")
                elif variable.index is not None:
                    names.append(f"{control.name}.{variable.name}-{variable.index}")
                else:
                    names.append(f"{control.name}.{variable.name}")
        return names

    @property
    def objective_names(self) -> list[str]:
        return [objective.name for objective in self.objective_functions]

    @property
    def constraint_names(self) -> list[str]:
        if self.output_constraints:
            return [constraint.name for constraint in self.output_constraints]
        return []

    @property
    def result_names(self) -> list[str]:
        objectives_names = [objective.name for objective in self.objective_functions]
        constraint_names = [
            constraint.name for constraint in (self.output_constraints or [])
        ]
        return objectives_names + constraint_names

    def to_dict(self) -> dict[str, Any]:
        the_dict = self.model_dump(exclude_none=True)

        if "config_path" in the_dict:
            the_dict["config_path"] = str(the_dict["config_path"])

        return the_dict

    @classmethod
    def with_defaults(cls, **kwargs):  # type: ignore
        """
        Creates an Everest config with default values. Useful for initializing a config
        without having to provide empty defaults.
        """
        defaults = {
            "controls": [
                {
                    "name": "default_group",
                    "type": "generic_control",
                    "initial_guess": 0.5,
                    "variables": [
                        {"name": "default_name", "min": 0, "max": 1},
                    ],
                }
            ],
            "objective_functions": [{"name": "default"}],
            "config_path": ".",
            "model": {"realizations": [0]},
        }

        return cls.model_validate({**defaults, **kwargs})

    @staticmethod
    def lint_config_dict(config: ConfigDict) -> list[ErrorDetails]:
        try:
            EverestConfig.model_validate(config)
            return []
        except ValidationError as err:
            return err.errors()

    @staticmethod
    def lint_config_dict_with_raise(config: ConfigDict) -> None:
        # Future work: Catch the validation error
        # and reformulate the pydantic ones to make them
        # more understandable
        EverestConfig.model_validate(config)

    @staticmethod
    def load_file(config_file: str) -> "EverestConfig":
        config_path = os.path.realpath(config_file)

        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"File not found: {config_path}")

        config_dict = yaml_file_to_substituted_config_dict(config_path)
        try:
            return EverestConfig.model_validate(config_dict)
        except ValidationError as error:
            exp = EverestValidationError()
            file_content = []
            with open(config_path, encoding="utf-8") as f:
                file_content = f.readlines()

            for e in error.errors(
                include_context=True, include_input=True, include_url=False
            ):
                if e["type"] in {"missing", "value_error"} or e["input"] is None:
                    exp.errors.append((e, None))
                else:
                    for index, line in enumerate(file_content):
                        if (pos := line.find(str(e["input"]))) != -1:
                            exp.errors.append((e, (index + 1, pos + 1)))
                            break
            raise exp from error

    @staticmethod
    def load_file_with_argparser(
        config_path: str, parser: ArgumentParser
    ) -> Optional["EverestConfig"]:
        try:
            return EverestConfig.load_file(config_path)
        except FileNotFoundError:
            parser.error(f"File not found: {config_path}")
        except YAMLError as e:
            parser.error(
                f"The config file: <{config_path}> contains invalid YAML syntax: {e!s}"
            )
        except EverestValidationError as e:
            parser.error(
                f"Loading config file <{config_path}> failed with:\n{format_errors(e)}"
            )
        except ValueError as e:
            parser.error(f"Loading config file <{config_path}> failed with: {e}")

    def dump(self, fname: str | None = None) -> str | None:
        """Write a config dict to file or return it if fname is None."""
        stripped_conf = self.to_dict()

        del stripped_conf["config_path"]

        yaml = YAML(typ="safe", pure=True)
        yaml.default_flow_style = False
        if fname is None:
            with StringIO() as sio:
                yaml.dump(stripped_conf, sio)
                return sio.getvalue()

        with open(fname, "w", encoding="utf-8") as out:
            yaml.dump(stripped_conf, out)

        return None

    @property
    def server_queue_system(self) -> QueueSystem:
        assert self.server is not None
        assert self.server.queue_system is not None
        return self.server.queue_system.name
