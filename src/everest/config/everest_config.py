import json
import logging
import os
import shutil
from argparse import ArgumentParser
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional, Protocol, no_type_check

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from ruamel.yaml import YAML, YAMLError
from typing_extensions import Annotated

from ert.config import ErtConfig
from everest.config.control_variable_config import ControlVariableGuessListConfig
from everest.config.install_template_config import InstallTemplateConfig
from everest.config.server_config import ServerConfig
from everest.config.validation_utils import (
    InstallDataContext,
    as_abs_path,
    check_for_duplicate_names,
    check_path_exists,
    check_writeable_path,
    format_errors,
    unique_items,
    validate_forward_model_configs,
)
from everest.jobs import script_names
from everest.util.forward_models import collect_forward_models

from ..config_file_loader import yaml_file_to_substituted_config_dict
from ..strings import (
    CERTIFICATE_DIR,
    DEFAULT_OUTPUT_DIR,
    DETACHED_NODE_DIR,
    HOSTFILE_NAME,
    OPTIMIZATION_LOG_DIR,
    OPTIMIZATION_OUTPUT_DIR,
    SERVER_STATUS,
    SESSION_DIR,
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
from .simulator_config import SimulatorConfig
from .well_config import WellConfig
from .workflow_config import WorkflowConfig

if TYPE_CHECKING:
    from pydantic_core import ErrorDetails


def _dummy_ert_config():
    site_config = ErtConfig.read_site_config()
    dummy_config = {"NUM_REALIZATIONS": 1, "ENSPATH": "."}
    dummy_config.update(site_config)
    return ErtConfig.with_plugins().from_dict(config_dict=dummy_config)


def get_system_installed_jobs():
    """Returns list of all system installed job names"""
    return list(_dummy_ert_config().installed_forward_model_steps.keys())


# Makes property.setter work
# Based on https://github.com/pydantic/pydantic/issues/1577#issuecomment-790506164
# We should use computed_property instead of this, when upgrading to pydantic 2.
class BaseModelWithPropertySupport(BaseModel):
    @no_type_check
    def __setattr__(self, name, value):
        """
        To be able to use properties with setters
        """
        try:
            getattr(self.__class__, name).fset(self, value)
        except AttributeError:
            super().__setattr__(name, value)


class HasName(Protocol):
    name: str


class EverestConfig(BaseModelWithPropertySupport):  # type: ignore
    controls: Annotated[List[ControlConfig], AfterValidator(unique_items)] = Field(
        description="""Defines a list of controls.
         Controls should have unique names each control defines
            a group of control variables
        """,
    )
    objective_functions: List[ObjectiveFunctionConfig] = Field(
        description="List of objective function specifications",
    )
    optimization: Optional[OptimizationConfig] = Field(
        default=OptimizationConfig(),
        description="Optimizer options",
    )
    model: Optional[ModelConfig] = Field(
        default=ModelConfig(),
        description="Configuration of the Everest model",
    )

    # It IS required but is currently used in a non-required manner by tests
    # Thus, it is to be made explicitly required as the other logic
    # is being rewritten
    environment: Optional[EnvironmentConfig] = Field(
        default=EnvironmentConfig(),
        description="The environment of Everest, specifies which folders are used "
        "for simulation and output, as well as the level of detail in Everest-logs",
    )
    wells: List[WellConfig] = Field(
        default_factory=lambda: [],
        description="A list of well configurations, all with unique names.",
    )
    definitions: Optional[dict] = Field(
        default_factory=lambda: {},
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
    input_constraints: Optional[List[InputConstraintConfig]] = Field(
        default=None, description="List of input constraints"
    )
    output_constraints: Optional[List[OutputConstraintConfig]] = Field(
        default=None, description="A list of output constraints with unique names."
    )
    install_jobs: Optional[List[InstallJobConfig]] = Field(
        default=None, description="A list of jobs to install"
    )
    install_workflow_jobs: Optional[List[InstallJobConfig]] = Field(
        default=None, description="A list of workflow jobs to install"
    )
    install_data: Optional[List[InstallDataConfig]] = Field(
        default=None,
        description="""A list of install data elements from the install_data config
        section. Each item marks what folders or paths need to be copied or linked
        in order for the evaluation jobs to run.""",
    )
    install_templates: Optional[List[InstallTemplateConfig]] = Field(
        default=None,
        description="""Allow the user to define the workflow establishing the model
        chain for the purpose of sensitivity analysis, enabling the relationship
        between sensitivity input variables and quantities of interests to be
        evaluated.
""",
    )
    server: Optional[ServerConfig] = Field(
        default=None,
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
    simulator: Optional[SimulatorConfig] = Field(
        default=None, description="Simulation settings"
    )
    forward_model: Optional[List[str]] = Field(
        default=None, description="List of jobs to run"
    )
    workflows: Optional[WorkflowConfig] = Field(
        default=None, description="Workflows to run during optimization"
    )
    export: Optional[ExportConfig] = Field(
        default=None,
        description="Settings to control the exports of a optimization run by everest.",
    )
    config_path: Path = Field()
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_install_job_sources(self):  # pylint: disable=E0213
        model = self.model
        config_path = self.config_path
        if not model or not config_path:
            return self

        errors = []
        config_dir = Path(config_path).parent
        realizations = model.realizations

        for install_jobs in (self.install_jobs, self.install_workflow_jobs):
            if not install_jobs:
                continue

            sources = [job.source for job in install_jobs]
            for source in sources:
                try:
                    check_path_exists(
                        path_source=source,
                        config_path=config_path,
                        realizations=realizations,
                    )
                except ValueError as e:
                    errors.append(str(e))
                abs_config_path = as_abs_path(source, str(config_dir))
                if not os.path.isfile(abs_config_path):
                    errors.append(f"Is not a file {abs_config_path}")
                    continue
                exec_path = None
                valid_jobfile = True
                with open(abs_config_path, "r", encoding="utf-8") as jobfile:
                    for line in jobfile:
                        if not line.startswith("EXECUTABLE"):
                            continue
                        data = line.strip().split()
                        if len(data) != 2:
                            valid_jobfile = False
                            continue
                        exec_path = data[1]
                        break

                if not valid_jobfile:
                    errors.append(f"malformed EXECUTABLE in {source}")
                    continue
                if exec_path is None:
                    errors.append(f"missing EXECUTABLE in {source}")
                    continue

                exec_relpath = os.path.join(os.path.dirname(abs_config_path), exec_path)
                if os.path.isfile(exec_relpath):
                    if os.access(exec_relpath, os.X_OK):
                        continue
                    # exec_relpath is a file, but not executable; flag and return False
                    errors.append(f"{exec_relpath} is not executable")
                    continue

                if not shutil.which(exec_path):
                    errors.append(f"No such executable {exec_path}")

        if len(errors) > 0:
            raise ValueError(errors)

        return self

    @model_validator(mode="after")
    def validate_forward_model_job_name_installed(self):  # pylint: disable=E0213
        install_jobs = self.install_jobs
        forward_model_jobs = self.forward_model
        if install_jobs is None:
            install_jobs = []
        if not forward_model_jobs:
            return self
        installed_jobs_name = [job.name for job in install_jobs]
        installed_jobs_name += list(script_names)  # default jobs
        installed_jobs_name += get_system_installed_jobs()  # system jobs
        installed_jobs_name += [job["name"] for job in collect_forward_models()]

        errors = []
        for fm_job in forward_model_jobs:
            job_name = fm_job.split()[0]
            if job_name not in installed_jobs_name:
                errors.append(f"unknown job {job_name}")

        if len(errors) > 0:  # Note: python3.11 ExceptionGroup will solve this nicely
            raise ValueError(errors)
        return self

    @model_validator(mode="after")
    def validate_workflow_name_installed(self):  # pylint: disable=E0213
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
    def validate_install_templates_unique_output_files(cls, install_templates):  # pylint: disable=E0213
        check_for_duplicate_names(
            [t.output_file for t in install_templates],
            "install_templates",
            "output_file",
        )
        return install_templates

    @model_validator(mode="after")
    def validate_install_templates_are_existing_files(self):
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
    def validate_cvar_nreals_interval(self):  # pylint: disable=E0213
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
    def validate_install_data_source_exists(self):
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
    def validate_model_data_file_exists(self):  # pylint: disable=E0213
        model = self.model
        if not model:
            return self
        config_path = self.config_path

        if model.data_file is not None:
            check_path_exists(model.data_file, config_path, model.realizations)

        return self

    @model_validator(mode="after")
    def validate_maintained_forward_models(self):
        install_data = self.install_data
        model = self.model
        realizations = model.realizations if model else [0]

        with InstallDataContext(install_data, self.config_path) as context:
            for realization in realizations:
                context.add_links_for_realization(realization)
                validate_forward_model_configs(self.forward_model, self.install_jobs)
        return self

    @model_validator(mode="after")
    # pylint: disable=E0213
    def validate_input_constraints_weight_definition(self):
        input_constraints = self.input_constraints
        if not input_constraints:
            return self

        controls = self.controls
        if controls is None:
            return self
        control_full_name = []
        errors = []
        for c in controls:
            for v in c.variables:
                if isinstance(v, ControlVariableGuessListConfig):
                    control_full_name.extend(
                        f"{c.name}.{v.name}-{index}"
                        for index, _ in enumerate(v.initial_guess, start=1)
                    )
                elif v.index is not None:
                    control_full_name.append(f"{c.name}.{v.name}-{v.index}")
                else:
                    control_full_name.append(f"{c.name}.{v.name}")

        for input_const in input_constraints:
            for key in input_const.weights:
                if key not in control_full_name:
                    errors.append(
                        f"Input control weight name {key} "
                        f"does not match any instance of "
                        f"control_name.variable_name-variable_index"
                    )

        if len(errors) > 0:  # Note: python3.11 ExceptionGroup will solve this nicely
            raise ValueError(errors)

        return self

    @model_validator(mode="after")
    def validate_variable_name_match_well_name(self):  # pylint: disable=E0213
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
    def validate_that_environment_sim_folder_is_writeable(self):
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
    def validate_unique_well_names(cls, wells: List[WellConfig]):
        check_for_duplicate_names([w.name for w in wells], "well", "name")
        return wells

    # pylint: disable=E0213
    @field_validator("output_constraints")
    @no_type_check
    @classmethod
    def validate_unique_output_constraint_names(
        cls, output_constraints: List[OutputConstraintConfig]
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

    @field_validator("objective_functions")
    @no_type_check
    @classmethod
    def validate_objective_function_aliases_valid(cls, functions):
        objective_names = [function.name for function in functions]

        aliases = [
            function.alias for function in functions if function.alias is not None
        ]
        for alias in aliases:
            if alias not in objective_names:
                raise ValueError(f"Invalid alias {alias}")
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
        level = self.environment.log_level if self.environment is not None else "info"

        if level is None:
            level = "info"

        levels = {
            "debug": logging.DEBUG,  # 10
            "info": logging.INFO,  # 20
            "warning": logging.WARNING,  # 30
            "error": logging.ERROR,  # 40
            "critical": logging.CRITICAL,  # 50
        }
        return levels.get(level.lower(), logging.INFO)

    @logging_level.setter
    def logging_level(
        self, level: Literal["debug", "info", "warning", "error", "critical"]
    ):
        env = self.environment
        assert env is not None
        env.log_level = level  # pylint:disable = E0237

    @property
    def config_directory(self) -> Optional[str]:
        if self.config_path is not None:
            return str(self.config_path.parent)

        return None

    @property
    def config_file(self) -> Optional[str]:
        if self.config_path is not None:
            return self.config_path.name
        return None

    @property
    def output_dir(self) -> Optional[str]:
        assert self.environment is not None
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
    def simulation_dir(self) -> Optional[str]:
        assert self.environment is not None
        path = self.environment.simulation_folder

        if os.path.isabs(path):
            return path

        cfgdir = self.output_dir
        if cfgdir is None:
            return path

        return os.path.join(cfgdir, path)

    def _get_output_subdirectory(self, subdirname: str):
        if self.output_dir is None:
            return None

        return os.path.join(os.path.abspath(self.output_dir), subdirname)

    @property
    def optimization_output_dir(self):
        """Return the path to folder with the optimization output"""
        return self._get_output_subdirectory(OPTIMIZATION_OUTPUT_DIR)

    @property
    def storage_dir(self):
        return self._get_output_subdirectory(STORAGE_DIR)

    @property
    def log_dir(self):
        return self._get_output_subdirectory(OPTIMIZATION_LOG_DIR)

    @property
    def detached_node_dir(self):
        return self._get_output_subdirectory(DETACHED_NODE_DIR)

    @property
    def session_dir(self):
        """Return path to the session directory containing information about the
        certificates and host information"""
        return os.path.join(self.detached_node_dir, SESSION_DIR)

    @property
    def certificate_dir(self):
        """Return the path to certificate folder"""
        return os.path.join(self.session_dir, CERTIFICATE_DIR)

    def get_server_url(self, server_info=None):
        """Return the url of the server.

        If server_info are given, the url is generated using that info. Otherwise
        server information are retrieved from the hostfile
        """
        if server_info is None:
            server_info = self.server_info

        url = f"https://{server_info['host']}:{server_info['port']}"
        return url

    @property
    def hostfile_path(self):
        return os.path.join(self.session_dir, HOSTFILE_NAME)

    @property
    def server_info(self):
        """Load server information from the hostfile"""
        host_file_path = self.hostfile_path

        with open(host_file_path, "r", encoding="utf-8") as f:
            json_string = f.read()

        data = json.loads(json_string)
        if set(data.keys()) != {"host", "port", "cert", "auth"}:
            raise RuntimeError("Malformed hostfile")

        return data

    @property
    def server_context(self):
        """Returns a tuple with
        - url of the server
        - path to the .cert file
        - password for the certificate file
        """

        return (
            self.get_server_url(self.server_info),
            self.server_info[CERTIFICATE_DIR],
            ("username", self.server_info["auth"]),
        )

    @property
    def export_path(self):
        """Returns the export file path. If not file name is provide the default
        export file name will have the same name as the config file, with the '.csv'
        extension."""

        export = self.export

        output_path = None
        if export is not None:
            output_path = export.csv_output_filepath

        if output_path is None:
            output_path = ""

        full_file_path = os.path.join(self.output_dir, output_path)
        if output_path:
            return full_file_path
        else:
            default_export_file = f"{os.path.splitext(self.config_file)[0]}.csv"
            return os.path.join(full_file_path, default_export_file)

    @property
    def everserver_status_path(self):
        """Returns path to the everest server status file"""
        return os.path.join(self.session_dir, SERVER_STATUS)

    def to_dict(self) -> dict:
        the_dict = self.model_dump(exclude_none=True)

        if "config_path" in the_dict:
            the_dict["config_path"] = str(the_dict["config_path"])

        return the_dict

    @classmethod
    def with_defaults(cls, **kwargs):
        """
        Creates an Everest config with default values. Useful for initializing a config
        without having to provide empty defaults.
        """
        defaults = {
            "controls": [],
            "objective_functions": [],
            "config_path": ".",
        }

        return EverestConfig.model_validate({**defaults, **kwargs})

    @staticmethod
    def lint_config_dict(config: dict) -> List["ErrorDetails"]:
        try:
            EverestConfig.model_validate(config)
            return []
        except ValidationError as err:
            return err.errors()

    @staticmethod
    def lint_config_dict_with_raise(config: dict):
        # Future work: Catch the validation error
        # and reformulate the pydantic ones to make them
        # more understandable
        EverestConfig.model_validate(config)

    @staticmethod
    def load_file(config_path: str) -> "EverestConfig":
        config_path = os.path.realpath(config_path)

        if not os.path.isfile(config_path):
            raise FileNotFoundError("File not found: {}".format(config_path))

        config_dict = yaml_file_to_substituted_config_dict(config_path)
        return EverestConfig.model_validate(config_dict)

    @staticmethod
    def load_file_with_argparser(
        config_path, parser: ArgumentParser
    ) -> Optional["EverestConfig"]:
        try:
            return EverestConfig.load_file(config_path)
        except FileNotFoundError:
            parser.error(f"File not found: {config_path}")
        except YAMLError as e:
            parser.error(
                f"The config file: <{config_path}> contains"
                f" invalid YAML syntax: {e!s}"
            )
        except ValidationError as e:
            parser.error(
                f"Loading config file <{config_path}> failed with:\n"
                f"{format_errors(e)}"
            )

    def dump(self, fname: Optional[str] = None) -> Optional[str]:
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
