# mypy: ignore-errors
import copy
import logging
import os
import pprint
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from functools import cached_property
from os import path
from pathlib import Path
from typing import Any, ClassVar, Self, no_type_check, overload

import polars as pl
from numpy.random import SeedSequence
from pydantic import BaseModel, Field, model_validator
from pydantic import ValidationError as PydanticValidationError

from ert.plugins import ErtPluginManager, fixtures_per_hook
from ert.substitutions import Substitutions

from ._design_matrix_validator import DesignMatrixValidator
from .analysis_config import AnalysisConfig
from .design_matrix import DESIGN_MATRIX_GROUP
from .ensemble_config import EnsembleConfig
from .forward_model_step import (
    ForwardModelJSON,
    ForwardModelStep,
    ForwardModelStepJSON,
    ForwardModelStepPlugin,
    ForwardModelStepValidationError,
    ForwardModelStepWarning,
)
from .gen_kw_config import GenKwConfig
from .model_config import ModelConfig
from .observation_vector import ObsVector
from .observations import EnkfObs
from .parse_arg_types_list import parse_arg_types_list
from .parsing import (
    ConfigDict,
    ConfigKeys,
    ConfigValidationError,
    ConfigWarning,
    ErrorInfo,
    ForwardModelStepKeys,
    HistorySource,
    HookRuntime,
    QueueSystemWithGeneric,
    init_forward_model_schema,
    init_site_config_schema,
    init_user_config_schema,
    parse_contents,
    read_file,
)
from .parsing.observations_parser import (
    GenObsValues,
    HistoryValues,
    ObservationConfigError,
    SummaryValues,
)
from .parsing.observations_parser import (
    parse_content as parse_observations,
)
from .queue_config import QueueConfig
from .workflow import Workflow
from .workflow_job import (
    ErtScriptLoadFailure,
    ErtScriptWorkflow,
    ExecutableWorkflow,
    _WorkflowJob,
    workflow_job_from_file,
)

logger = logging.getLogger(__name__)

EMPTY_LINES = re.compile(r"\n[\s\n]*\n")

ECL_BASE_DEPRECATION_MSG = (
    "Substitution template <ECL_BASE> is deprecated and "
    "will be removed in the future. Please use <ECLBASE> instead."
)


def site_config_location() -> str | None:
    if "ERT_SITE_CONFIG" in os.environ:
        return os.environ["ERT_SITE_CONFIG"]
    return None


def _seed_sequence(seed: int | None) -> int:
    # Set up RNG
    if seed is None:
        int_seed = SeedSequence().entropy
        logger.info(
            "To repeat this experiment, "
            "add the following random seed to your config file:\n"
            f"RANDOM_SEED {int_seed}"
        )
    else:
        int_seed = seed
    assert isinstance(int_seed, int)
    return int_seed


def _read_time_map(file_contents: str) -> list[datetime]:
    def str_to_datetime(date_str: str) -> datetime:
        try:
            return datetime.fromisoformat(date_str)
        except ValueError:
            logger.warning(
                "DD/MM/YYYY date format is deprecated"
                ", please use ISO date format YYYY-MM-DD."
            )
            return datetime.strptime(date_str, "%d/%m/%Y")

    dates = []
    for line in file_contents.splitlines():
        dates.append(str_to_datetime(line.strip()))
    return dates


def create_forward_model_json(
    context: dict[str, str],
    forward_model_steps: list[ForwardModelStep],
    run_id: str | None,
    iens: int = 0,
    itr: int = 0,
    user_config_file: str | None = "",
    env_vars: dict[str, str] | None = None,
    env_pr_fm_step: dict[str, dict[str, Any]] | None = None,
    skip_pre_experiment_validation: bool = False,
) -> ForwardModelJSON:
    if env_vars is None:
        env_vars = {}
    if env_pr_fm_step is None:
        env_pr_fm_step = {}

    context_substitutions = Substitutions(context)

    class Substituter:
        def __init__(self, fm_step) -> None:
            fm_step_args = ",".join(
                [f"{key}={value}" for key, value in fm_step.private_args.items()]
            )
            fm_step_description = f"{fm_step.name}({fm_step_args})"
            self.substitution_context_hint = (
                f"parsing forward model step `FORWARD_MODEL {fm_step_description}` - "
                "reconstructed, with defines applied during parsing"
            )
            self.copy_private_args = Substitutions(
                {
                    key: context_substitutions.substitute_real_iter(val, iens, itr)
                    for key, val in fm_step.private_args.items()
                }
            )

        @overload
        def substitute(self, string: str) -> str: ...

        @overload
        def substitute(self, string: None) -> None: ...

        def substitute(self, string):
            if string is None:
                return string
            string = self.copy_private_args.substitute(
                string, self.substitution_context_hint, 1, warn_max_iter=False
            )
            return context_substitutions.substitute_real_iter(string, iens, itr)

        def filter_env_dict(self, env_dict: dict[str, str]) -> dict[str, str] | None:
            substituted_dict = {}
            for key, value in env_dict.items():
                substituted_key = self.substitute(key)
                substituted_value = self.substitute(value)
                if substituted_value is None:
                    substituted_dict[substituted_key] = None
                elif not substituted_value:
                    substituted_dict[substituted_key] = ""
                elif not (substituted_value[0] == "<" and substituted_value[-1] == ">"):
                    # Remove values containing "<XXX>". These are expected to be
                    # replaced by substitute, but were not.
                    substituted_dict[substituted_key] = substituted_value
                else:
                    logger.warning(
                        f"Environment variable {substituted_key} skipped due to"
                        f" unmatched define {substituted_value}",
                    )
            # Its expected that empty dicts be replaced with "null"
            # in jobs.json
            if not substituted_dict:
                return None
            return substituted_dict

    def handle_default(fm_step: ForwardModelStep, arg: str) -> str:
        return fm_step.default_mapping.get(arg, arg)

    for fm_step in forward_model_steps:
        for key, val in fm_step.private_args.items():
            if key in context and key != val and context[key] != val:
                logger.info(
                    f"Private arg '{key}':'{val}' chosen over"
                    f" global '{context[key]}' in forward model step {fm_step.name}"
                )
    config_file_path = Path(user_config_file) if user_config_file is not None else None
    config_path = str(config_file_path.parent) if config_file_path else ""
    config_file = str(config_file_path.name) if config_file_path else ""

    job_list_errors = []
    job_list: list[ForwardModelStepJSON] = []
    for idx, fm_step in enumerate(forward_model_steps):
        substituter = Substituter(fm_step)
        fm_step_json: ForwardModelStepJSON = {
            "name": substituter.substitute(fm_step.name),
            "executable": substituter.substitute(fm_step.executable),
            "target_file": substituter.substitute(fm_step.target_file),
            "error_file": substituter.substitute(fm_step.error_file),
            "start_file": substituter.substitute(fm_step.start_file),
            "stdout": (
                substituter.substitute(fm_step.stdout_file) + f".{idx}"
                if fm_step.stdout_file
                else None
            ),
            "stderr": (
                substituter.substitute(fm_step.stderr_file) + f".{idx}"
                if fm_step.stderr_file
                else None
            ),
            "stdin": substituter.substitute(fm_step.stdin_file),
            "argList": [
                handle_default(fm_step, substituter.substitute(arg))
                for arg in fm_step.arglist
            ],
            "environment": substituter.filter_env_dict(
                dict(
                    **{
                        key: value
                        for key, value in env_pr_fm_step.get(fm_step.name, {}).items()
                        # Plugin settings can not override anything:
                        if key not in env_vars and key not in fm_step.environment
                    },
                    **fm_step.environment,
                )
            ),
            "max_running_minutes": fm_step.max_running_minutes,
        }

        try:
            if not skip_pre_experiment_validation:
                fm_step_json = fm_step.validate_pre_realization_run(fm_step_json)
        except ForwardModelStepValidationError as exc:
            job_list_errors.append(
                ErrorInfo(
                    message=f"Validation failed for "
                    f"forward model step {fm_step.name}: {exc!s}"
                ).set_context(fm_step.name)
            )

        job_list.append(fm_step_json)

    if job_list_errors:
        raise ConfigValidationError.from_collected(job_list_errors)

    return {
        "global_environment": env_vars,
        "config_path": config_path,
        "config_file": config_file,
        "jobList": job_list,
        "run_id": run_id,
        "ert_pid": str(os.getpid()),
    }


def check_non_utf_chars(file_path: str) -> None:
    try:
        with open(file_path, encoding="utf-8") as f:
            f.read()
    except UnicodeDecodeError as e:
        error_words = str(e).split(" ")
        hex_str = error_words[error_words.index("byte") + 1]
        try:
            unknown_char = chr(int(hex_str, 16))
        except ValueError as ve:
            unknown_char = f"hex:{hex_str}"
            raise ConfigValidationError(
                f"Unsupported non UTF-8 character {unknown_char!r} "
                f"found in file: {file_path!r}",
                config_file=file_path,
            ) from ve
        raise ConfigValidationError(
            f"Unsupported non UTF-8 character {unknown_char!r} "
            f"found in file: {file_path!r}",
            config_file=file_path,
        ) from e


def read_templates(config_dict) -> list[tuple[str, str]]:
    templates = []
    if ConfigKeys.DATA_FILE in config_dict and ConfigKeys.ECLBASE in config_dict:
        source_file = config_dict[ConfigKeys.DATA_FILE]
        target_file = config_dict[ConfigKeys.ECLBASE].replace("%d", "<IENS>") + ".DATA"
        check_non_utf_chars(source_file)
        templates.append([source_file, target_file])

    for template in config_dict.get(ConfigKeys.RUN_TEMPLATE, []):
        if template[1].startswith("<ECL_BASE>"):
            ConfigWarning.warn(ECL_BASE_DEPRECATION_MSG)
        if (
            ConfigKeys.ECLBASE in config_dict
            and (
                template[1].startswith(config_dict[ConfigKeys.ECLBASE])
                or template[1].startswith("<ECLBASE>")
                or template[1].startswith("<ECL_BASE>")
            )
            and ConfigKeys.NUM_CPU not in config_dict
        ):
            ConfigWarning.warn(
                "Use DATA_FILE instead of RUN_TEMPLATE for "
                "templating the Eclipse/Flow DATA file. "
                "This ensures correct parsing of NUM_CPU. "
                "Alternatively set NUM_CPU explicitly and ensure "
                "it is synced with your DATA file."
            )
        templates.append(template)
    templates.extend(EnsembleConfig.get_gen_kw_templates(config_dict))
    return templates


def workflow_jobs_from_dict(
    content_dict: ConfigDict,
    installed_workflows: dict[str, ErtScriptWorkflow] | None = None,
) -> dict[str, _WorkflowJob]:
    workflow_job_info = content_dict.get(ConfigKeys.LOAD_WORKFLOW_JOB, [])
    workflow_job_dir_info = content_dict.get(ConfigKeys.WORKFLOW_JOB_DIRECTORY, [])

    workflow_jobs = copy.copy(installed_workflows) if installed_workflows else {}

    errors = []

    for workflow_job in workflow_job_info:
        try:
            # workflow_job_from_file only throws error if a
            # non-readable file is provided.
            # Non-existing files are caught by the new parser
            new_job = workflow_job_from_file(
                config_file=workflow_job[0],
                name=None if len(workflow_job) == 1 else workflow_job[1],
            )
            name = new_job.name
            if name in workflow_jobs:
                prop = (
                    new_job.executable
                    if isinstance(new_job, ExecutableWorkflow)
                    else new_job.ert_script
                )
                old_prop = (
                    workflow_jobs[name].executable
                    if isinstance(workflow_jobs[name], ExecutableWorkflow)
                    else workflow_jobs[name].ert_script
                )
                ConfigWarning.warn(
                    f"Duplicate workflow jobs with name {name!r}, choosing "
                    f"{prop!r} over "
                    f"{old_prop!r}",
                    name,
                )
            workflow_jobs[name] = new_job
        except ErtScriptLoadFailure as err:
            ConfigWarning.warn(
                f"Loading workflow job {workflow_job[0]!r}"
                f" failed with '{err}'. It will not be loaded.",
                workflow_job[0],
            )
        except ConfigValidationError as err:
            errors.append(ErrorInfo(message=str(err)).set_context(workflow_job[0]))

    for job_path in workflow_job_dir_info:
        for file_name in _get_files_in_directory(job_path, errors):
            try:
                new_job = workflow_job_from_file(config_file=file_name)
                name = new_job.name
                if name in workflow_jobs:
                    ConfigWarning.warn(
                        f"Duplicate workflow jobs with name {name!r}, choosing "
                        f"{new_job.executable or new_job.ert_script!r} over "
                        f"{workflow_jobs[name].executable or workflow_jobs[name].ert_script!r}",  # noqa: E501
                        name,
                    )
                workflow_jobs[name] = new_job
            except ErtScriptLoadFailure as err:
                ConfigWarning.warn(
                    f"Loading workflow job {file_name!r}"
                    f" failed with '{err}'. It will not be loaded.",
                    file_name,
                )
            except ConfigValidationError as err:
                errors.append(ErrorInfo(message=str(err)).set_context(job_path))
    if errors:
        raise ConfigValidationError.from_collected(errors)

    return workflow_jobs


def create_and_hook_workflows(
    content_dict: ConfigDict,
    workflow_jobs: dict[str, _WorkflowJob],
    substitutions: dict[str, str],
) -> tuple[dict[str, Workflow], defaultdict[HookRuntime, list[Workflow]]]:
    hook_workflow_info = content_dict.get(ConfigKeys.HOOK_WORKFLOW, [])
    workflow_info = content_dict.get(ConfigKeys.LOAD_WORKFLOW, [])

    workflows = {}
    hooked_workflows = defaultdict(list)

    errors = []

    for work in workflow_info:
        filename = path.basename(work[0]) if len(work) == 1 else work[1]
        try:
            existed = filename in workflows
            workflow = Workflow.from_file(
                work[0],
                substitutions,
                workflow_jobs,
            )
            for job, args in workflow:
                if isinstance(job, ErtScriptWorkflow):
                    try:
                        job.ert_script.validate(args)
                    except ConfigValidationError as err:
                        errors.append(ErrorInfo(message=str(err)).set_context(work[0]))
                        continue
            workflows[filename] = workflow
            if existed:
                ConfigWarning.warn(f"Workflow {filename!r} was added twice", work[0])
        except ConfigValidationError as err:
            ConfigWarning.warn(
                f"Encountered the following error(s) while "
                f"reading workflow {filename!r}. It will not be loaded: "
                + err.cli_message(),
                work[0],
            )

    for hook_name, mode in hook_workflow_info:
        if hook_name not in workflows:
            errors.append(
                ErrorInfo(
                    message="Cannot setup hook for non-existing"
                    f" job name {hook_name!r}",
                ).set_context(hook_name)
            )
            continue

        wf = workflows[hook_name]
        available_fixtures = fixtures_per_hook[mode]
        for job, _ in wf.cmd_list:
            if not hasattr(job, "ert_script") or job.ert_script is None:
                continue

            ert_script_instance = job.ert_script()
            requested_fixtures = ert_script_instance.requested_fixtures

            # Look for requested fixtures that are not available for the given
            # mode
            missing_fixtures = requested_fixtures - available_fixtures

            if missing_fixtures:
                ok_modes = [
                    m
                    for m in HookRuntime
                    if not requested_fixtures - fixtures_per_hook[m]
                ]

                message_start = (
                    f"Workflow job {job.name} .run function expected "
                    f"fixtures: {missing_fixtures}, which are not available "
                    f"in the fixtures for the runtime {mode}: {available_fixtures}. "
                )
                message_end = (
                    f"It would work in these runtimes: {', '.join(map(str, ok_modes))}"
                    if len(ok_modes) > 0
                    else "This fixture is not available in any of the runtimes."
                )

                errors.append(
                    ErrorInfo(message=message_start + message_end).set_context(
                        hook_name
                    )
                )

        hooked_workflows[mode].append(workflows[hook_name])

    if errors:
        raise ConfigValidationError.from_collected(errors)

    return workflows, hooked_workflows


@staticmethod
def workflows_from_dict(
    content_dict: ConfigDict,
    substitutions: dict[str, str],
    installed_workflows: Mapping[str, _WorkflowJob] | None = None,
) -> tuple[
    dict[str, _WorkflowJob],
    dict[str, Workflow],
    defaultdict[HookRuntime, list[Workflow]],
]:
    workflow_jobs = copy.copy(installed_workflows) if installed_workflows else {}
    workflow_jobs = workflow_jobs_from_dict(content_dict, workflow_jobs)
    workflows, hooked_workflows = create_and_hook_workflows(
        content_dict, workflow_jobs, substitutions
    )
    return workflow_jobs, workflows, hooked_workflows


def installed_forward_model_steps_from_dict(config_dict) -> dict[str, ForwardModelStep]:
    errors = []
    fm_steps = {}
    for name, (fm_step_config_file, config_contents) in config_dict.get(
        ConfigKeys.INSTALL_JOB, []
    ):
        fm_step_config_file = path.abspath(fm_step_config_file)
        try:
            new_fm_step = _forward_model_step_from_config_contents(
                config_contents,
                name=name,
                config_file=fm_step_config_file,
            )
        except ConfigValidationError as e:
            errors.append(e)
            continue
        if name in fm_steps:
            ConfigWarning.warn(
                f"Duplicate forward model step with name {name!r}, choosing "
                f"{fm_step_config_file!r} over {fm_steps[name].executable!r}",
                name,
            )
        fm_steps[name] = new_fm_step

    for fm_step_path in config_dict.get(ConfigKeys.INSTALL_JOB_DIRECTORY, []):
        for file_name in _get_files_in_directory(fm_step_path, errors):
            if not path.isfile(file_name):
                continue
            try:
                config_contents = read_file(file_name)
                new_fm_step = _forward_model_step_from_config_contents(
                    config_contents, config_file=file_name
                )
            except ConfigValidationError as e:
                errors.append(e)
                continue
            name = new_fm_step.name
            if name in fm_steps:
                ConfigWarning.warn(
                    f"Duplicate forward model step with name {name!r}, "
                    f"choosing {file_name!r} over {fm_steps[name].executable!r}",
                    name,
                )
            fm_steps[name] = new_fm_step

    if errors:
        raise ConfigValidationError.from_collected(errors)
    return fm_steps


def create_list_of_forward_model_steps_to_run(
    installed_steps: dict[str, ForwardModelStep],
    substitutions: dict[str, str],
    config_dict: dict,
    preinstalled_forward_model_steps: dict[str, ForwardModelStep],
    env_pr_fm_step: dict[str, dict[str, Any]],
) -> list[ForwardModelStep]:
    errors = []
    fm_steps: list[ForwardModelStep] = []

    substituter = Substitutions(substitutions)
    env_vars = {}
    for key, val in config_dict.get("SETENV", []):
        env_vars[key] = substituter.substitute(val)

    for fm_step_description in config_dict.get(ConfigKeys.FORWARD_MODEL, []):
        if len(fm_step_description) > 1:
            unsubstituted_step_name, args = fm_step_description
        else:
            unsubstituted_step_name = fm_step_description[0]
            args = []
        fm_step_name = substituter.substitute(unsubstituted_step_name)
        try:
            fm_step = copy.deepcopy(installed_steps[fm_step_name])

            # Preserve as ContextString
            fm_step.name = fm_step_name
        except KeyError:
            errors.append(
                ConfigValidationError.with_context(
                    f"Could not find forward model step {fm_step_name!r} in list "
                    "of installed forward model steps: "
                    f"{list(installed_steps.keys())!r}",
                    fm_step_name,
                )
            )
            continue

        fm_step.private_args = {}
        for arg in args:
            match arg:
                case key, val:
                    fm_step.private_args[key] = val
                case val:
                    fm_step.arglist.append(val)

        try:
            fm_step.check_required_keywords()
        except ConfigValidationError as err:
            errors.append(err)
            continue
        fm_steps.append(fm_step)

    dm_validator = DesignMatrixValidator()
    for fm_step in fm_steps:
        if fm_step.name == "DESIGN2PARAMS":
            dm_validator.validate_design_matrix(fm_step.private_args)

        if fm_step.name in preinstalled_forward_model_steps:
            if "<ECL_BASE>" in str(fm_step):
                ConfigWarning.warn(
                    ECL_BASE_DEPRECATION_MSG,
                    context=fm_step.name,
                )
            try:
                substituted_json = create_forward_model_json(
                    run_id=None,
                    context=substitutions,
                    forward_model_steps=[fm_step],
                    skip_pre_experiment_validation=True,
                    env_vars=env_vars,
                    env_pr_fm_step=env_pr_fm_step,
                )
                fm_json_for_validation = substituted_json["jobList"][0]
                fm_json_for_validation["environment"] = {
                    **substituted_json["global_environment"],
                    **fm_json_for_validation["environment"],
                }
                fm_step.validate_pre_experiment(fm_json_for_validation)
            except ForwardModelStepValidationError as err:
                errors.append(
                    ConfigValidationError.with_context(
                        f"Forward model step pre-experiment validation failed: {err!s}",
                        context=fm_step.name,
                    ),
                )
            except ForwardModelStepWarning as err:
                ConfigWarning.warn(
                    f"Forward model step validation: {err!s}",
                    context=fm_step.name,
                )

            except Exception as e:  # type: ignore
                ConfigWarning.warn(
                    f"Unexpected plugin forward model exception: {e!s}",
                    context=fm_step.name,
                )
    dm_validator.validate_design_matrix_merge()

    if errors:
        raise ConfigValidationError.from_collected(errors)

    return fm_steps


RESERVED_KEYWORDS = ["realization", "IENS", "ITER"]


class ErtConfig(BaseModel):
    DEFAULT_ENSPATH: ClassVar[str] = "storage"
    DEFAULT_RUNPATH_FILE: ClassVar[str] = ".ert_runpath_list"
    PREINSTALLED_FORWARD_MODEL_STEPS: ClassVar[dict[str, ForwardModelStep]] = {}
    PREINSTALLED_WORKFLOWS: ClassVar[dict[str, ErtScriptWorkflow]] = {}
    ENV_PR_FM_STEP: ClassVar[dict[str, dict[str, Any]]] = {}
    ACTIVATE_SCRIPT: ClassVar[str | None] = None
    RESERVED_KEYWORDS: ClassVar[list[str]] = RESERVED_KEYWORDS

    substitutions: dict[str, str] = Field(default_factory=dict)
    ensemble_config: EnsembleConfig = Field(default_factory=EnsembleConfig)
    ens_path: str = DEFAULT_ENSPATH
    env_vars: dict[str, str] = Field(default_factory=dict)
    random_seed: int = Field(default_factory=lambda: _seed_sequence(None))
    analysis_config: AnalysisConfig = Field(default_factory=AnalysisConfig)
    queue_config: QueueConfig = Field(default_factory=QueueConfig)
    workflow_jobs: dict[str, _WorkflowJob] = Field(default_factory=dict)
    workflows: dict[str, Workflow] = Field(default_factory=dict)
    hooked_workflows: defaultdict[HookRuntime, list[Workflow]] = Field(
        default_factory=lambda: defaultdict(list)
    )
    runpath_file: Path = Path(DEFAULT_RUNPATH_FILE)

    ert_templates: list[tuple[str, str]] = Field(default_factory=list)
    installed_forward_model_steps: dict[str, ForwardModelStep] = Field(
        default_factory=dict
    )

    forward_model_steps: list[ForwardModelStep] = Field(default_factory=list)
    runpath_config: ModelConfig = Field(default_factory=ModelConfig)
    user_config_file: str = "no_config"
    config_path: str = Field(init=False, default="")
    observation_config: list[
        tuple[str, HistoryValues | SummaryValues | GenObsValues]
    ] = Field(default_factory=list)
    enkf_obs: EnkfObs = Field(default_factory=EnkfObs)

    @model_validator(mode="after")
    def set_fields(self):
        self.config_path = (
            path.dirname(path.abspath(self.user_config_file))
            if self.user_config_file
            else os.getcwd()
        )
        return self

    @model_validator(mode="after")
    def validate_genkw_parameter_name_overlap(self):
        overlapping_parameter_names = [
            parameter_name
            for parameter_name in self.ensemble_config.get_all_gen_kw_parameter_names()
            if f"<{parameter_name}>" in self.substitutions
            or parameter_name in ErtConfig.RESERVED_KEYWORDS
        ]
        if overlapping_parameter_names:
            raise ConfigValidationError(
                f"Found reserved parameter name(s): "
                f"{', '.join(overlapping_parameter_names)}. The names are already in "
                "use as magic strings or defined in the user config."
            )
        return self

    @model_validator(mode="after")
    def validate_dm_parameter_name_overlap(self):
        if not self.analysis_config.design_matrix:
            return self
        dm_param_config = self.analysis_config.design_matrix.parameter_configuration
        overlapping_parameter_names = [
            parameter_definition.name
            for parameter_definition in dm_param_config.transform_function_definitions
            if f"<{parameter_definition.name}>" in self.substitutions
            or parameter_definition.name in ErtConfig.RESERVED_KEYWORDS
        ]
        if overlapping_parameter_names:
            raise ConfigValidationError(
                f"Found reserved parameter name(s): "
                f"{', '.join(overlapping_parameter_names)}. The names are already in "
                "use as magic strings or defined in the user config."
            )
        return self

    @property
    def observations(self) -> dict[str, pl.DataFrame]:
        return self.enkf_obs.datasets

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ErtConfig):
            return False

        for attr in vars(self):
            if attr == "observations":
                if self.observations.keys() != other.observations.keys():
                    return False

                if not all(
                    self.observations[k].equals(other.observations[k])
                    for k in self.observations
                ):
                    return False

                continue

            if getattr(self, attr) != getattr(other, attr):
                return False

        return True

    @staticmethod
    def with_plugins(
        forward_model_step_classes: list[type[ForwardModelStepPlugin]] | None = None,
        env_pr_fm_step: dict[str, dict[str, Any]] | None = None,
    ) -> type["ErtConfig"]:
        pm = ErtPluginManager()
        if forward_model_step_classes is None:
            forward_model_step_classes = pm.forward_model_steps

        preinstalled_fm_steps: dict[str, ForwardModelStepPlugin] = {}
        for fm_step_subclass in forward_model_step_classes:
            fm_step = fm_step_subclass()
            preinstalled_fm_steps[fm_step.name] = fm_step

        if env_pr_fm_step is None:
            env_pr_fm_step = uppercase_subkeys_and_stringify_subvalues(
                pm.get_forward_model_configuration()
            )

        class ErtConfigWithPlugins(ErtConfig):
            PREINSTALLED_FORWARD_MODEL_STEPS: ClassVar[
                dict[str, ForwardModelStepPlugin]
            ] = preinstalled_fm_steps
            PREINSTALLED_WORKFLOWS = pm.get_ertscript_workflows().get_workflows()
            ENV_PR_FM_STEP: ClassVar[dict[str, dict[str, Any]]] = env_pr_fm_step
            ACTIVATE_SCRIPT = pm.activate_script()

        from datetime import datetime  # noqa

        ErtConfigWithPlugins.model_rebuild()
        assert issubclass(ErtConfigWithPlugins, ErtConfig)
        return ErtConfigWithPlugins

    @classmethod
    def from_file(cls, user_config_file: str) -> Self:
        """
        Reads the given :ref:`User Config File<List of keywords>` and the
        `Site wide configuration` and returns an ErtConfig containing the
        configured values specified in those files.

        Raises:
            ConfigValidationError: Signals one or more incorrectly configured
            value(s) that the user needs to fix before ert can run.


        Warnings will be issued with :python:`warnings.warn(category=ConfigWarning)`
        when the user should be notified with non-fatal configuration problems.
        """
        user_config_contents = read_file(user_config_file)
        cls._log_config_file(user_config_file, user_config_contents)
        site_config_file = site_config_location()
        user_config_dict = cls._config_dict_from_contents(
            user_config_contents,
            read_file(site_config_file) if site_config_file else None,
            user_config_file,
            site_config_file,
        )
        cls._log_config_dict(user_config_dict)
        return cls.from_dict(user_config_dict)

    @classmethod
    def _config_dict_from_contents(
        cls,
        user_config_contents: str,
        site_config_contents: str | None,
        config_file_name: str,
        site_config_name: str,
    ) -> ConfigDict:
        site_config_dict = (
            parse_contents(
                site_config_contents,
                file_name=site_config_name,
                schema=init_site_config_schema(),
            )
            if site_config_contents
            else ConfigDict()
        )
        user_config_dict = cls._read_user_config_and_apply_site_config(
            user_config_contents,
            config_file_name,
            site_config_dict,
        )
        config_dir = path.abspath(path.dirname(config_file_name))
        cls.apply_config_content_defaults(user_config_dict, config_dir)
        return user_config_dict

    @classmethod
    def from_file_contents(
        cls,
        user_config_contents: str,
        site_config_contents: str = "QUEUE_SYSTEM LOCAL\n",
        config_file_name="./config.ert",
        site_config_name="site_config.ert",
    ) -> Self:
        return cls.from_dict(
            cls._config_dict_from_contents(
                user_config_contents,
                site_config_contents,
                config_file_name,
                site_config_name,
            )
        )

    @classmethod
    def from_dict(cls, config_dict) -> Self:
        substitutions = _substitutions_from_dict(config_dict)
        runpath_file = config_dict.get(
            ConfigKeys.RUNPATH_FILE, ErtConfig.DEFAULT_RUNPATH_FILE
        )
        substitutions["<RUNPATH_FILE>"] = runpath_file
        config_dir = substitutions.get("<CONFIG_PATH>", "")
        config_file = substitutions.get("<CONFIG_FILE>", "no_config")
        config_file_path = path.join(config_dir, config_file)

        errors = cls._validate_dict(config_dict, config_file)

        if errors:
            raise ConfigValidationError.from_collected(errors)

        workflow_jobs = {}
        workflows = {}
        hooked_workflows = {}
        installed_forward_model_steps = {}
        model_config = None

        try:
            model_config = ModelConfig.from_dict(config_dict)
            runpath = model_config.runpath_format_string
            eclbase = model_config.eclbase_format_string
            substitutions["<RUNPATH>"] = runpath
            substitutions["<ECL_BASE>"] = eclbase
            substitutions["<ECLBASE>"] = eclbase
        except ConfigValidationError as e:
            errors.append(e)
        except PydanticValidationError as err:
            errors.append(ConfigValidationError.from_pydantic(err))

        try:
            workflow_jobs, workflows, hooked_workflows = workflows_from_dict(
                config_dict, substitutions, cls.PREINSTALLED_WORKFLOWS
            )
        except ConfigValidationError as e:
            errors.append(e)

        try:
            installed_forward_model_steps = copy.deepcopy(
                cls.PREINSTALLED_FORWARD_MODEL_STEPS
            )

            installed_forward_model_steps.update(
                installed_forward_model_steps_from_dict(config_dict)
            )

        except ConfigValidationError as e:
            errors.append(e)

        try:
            if cls.ACTIVATE_SCRIPT:
                if "QUEUE_OPTION" not in config_dict:
                    config_dict["QUEUE_OPTION"] = []
                config_dict["QUEUE_OPTION"].append(
                    [
                        QueueSystemWithGeneric.GENERIC,
                        "ACTIVATE_SCRIPT",
                        cls.ACTIVATE_SCRIPT,
                    ]
                )
            queue_config = QueueConfig.from_dict(config_dict)

            substitutions["<NUM_CPU>"] = str(queue_config.queue_options.num_cpu)

        except ConfigValidationError as err:
            errors.append(err)

        obs_config_args = config_dict.get(ConfigKeys.OBS_CONFIG)
        obs_configs = []
        try:
            if obs_config_args:
                obs_config_file, obs_config_file_contents = obs_config_args
                obs_configs = parse_observations(
                    obs_config_file_contents, obs_config_file
                )
                if not obs_configs:
                    raise ObservationConfigError.with_context(
                        f"Empty observations file: {obs_config_file}",
                        obs_config_file,
                    )
        except ObservationConfigError as err:
            errors.append(err)

        try:
            if obs_configs:
                summary_obs = {
                    obs[1].key
                    for obs in obs_configs
                    if isinstance(obs[1], HistoryValues | SummaryValues)
                }
                if summary_obs:
                    summary_keys = ErtConfig._read_summary_keys(config_dict)
                    config_dict[ConfigKeys.SUMMARY] = [summary_keys] + [
                        [key] for key in summary_obs if key not in summary_keys
                    ]
            ensemble_config = EnsembleConfig.from_dict(config_dict=config_dict)
            time_map = None
            if time_map_args := config_dict.get(ConfigKeys.TIME_MAP):
                time_map_file, time_map_contents = time_map_args
                try:
                    time_map = _read_time_map(time_map_contents)
                except ValueError as err:
                    raise ConfigValidationError.with_context(
                        f"Could not read timemap file {time_map_file}: {err}",
                        time_map_file,
                    ) from err
            observations = cls._create_observations(
                obs_configs,
                ensemble_config,
                time_map,
                config_dict.get(
                    ConfigKeys.HISTORY_SOURCE, HistorySource.REFCASE_HISTORY
                ),
            )
        except ConfigValidationError as err:
            errors.append(err)
        except PydanticValidationError as err:
            errors.append(ConfigValidationError.from_pydantic(err))

        try:
            analysis_config = AnalysisConfig.from_dict(config_dict)
        except ConfigValidationError as err:
            errors.append(err)

        if errors:
            raise ConfigValidationError.from_collected(errors)

        if dm := analysis_config.design_matrix:
            dm_errors = []
            dm_params = {
                x.name
                for x in dm.parameter_configuration.transform_function_definitions
            }
            for group_name, config in ensemble_config.parameter_configs.items():
                if not isinstance(config, GenKwConfig):
                    continue
                group_params = {x.name for x in config.transform_function_definitions}
                if group_name == DESIGN_MATRIX_GROUP:
                    dm_errors.append(
                        ConfigValidationError(
                            f"Cannot have GEN_KW with group name {DESIGN_MATRIX_GROUP} "
                            "when using DESIGN_MATRIX keyword."
                        )
                    )
                if dm_params == group_params:
                    ConfigWarning.warn(
                        f"Parameters {group_params} from GEN_KW group '{group_name}' "
                        "will be overridden by design matrix. This will cause "
                        "updates to be turned off for these parameters."
                    )
                elif intersection := dm_params & group_params:
                    dm_errors.append(
                        ConfigValidationError(
                            "Only full overlaps of design matrix and "
                            "one genkw group are supported.\n"
                            f"design matrix parameters: {dm_params}\n"
                            f"parameters in genkw group <{group_name}>: "
                            f"{group_params}\n"
                            f"overlap between them: {intersection}"
                        )
                    )
            if dm_errors:
                raise ConfigValidationError.from_collected(dm_errors)

        env_vars = {}
        substituter = Substitutions(substitutions)
        for key, val in config_dict.get("SETENV", []):
            env_vars[key] = substituter.substitute(val)
        try:
            return cls(
                substitutions=substitutions,
                ensemble_config=ensemble_config,
                ens_path=config_dict.get(ConfigKeys.ENSPATH, ErtConfig.DEFAULT_ENSPATH),
                env_vars=env_vars,
                random_seed=_seed_sequence(config_dict.get(ConfigKeys.RANDOM_SEED)),
                analysis_config=analysis_config,
                queue_config=queue_config,
                workflow_jobs=workflow_jobs,
                workflows=workflows,
                hooked_workflows=hooked_workflows,
                runpath_file=Path(runpath_file),
                ert_templates=read_templates(config_dict),
                installed_forward_model_steps=installed_forward_model_steps,
                forward_model_steps=cls._create_list_of_forward_model_steps_to_run(
                    installed_forward_model_steps,
                    substitutions,
                    config_dict,
                ),
                runpath_config=model_config,
                user_config_file=config_file_path,
                observation_config=obs_configs,
                enkf_obs=observations,
            )
        except PydanticValidationError as err:
            raise ConfigValidationError.from_pydantic(err) from err

    @classmethod
    def _create_list_of_forward_model_steps_to_run(
        cls,
        installed_steps: dict[str, ForwardModelStep],
        substitutions: dict[str, str],
        config_dict: dict,
    ) -> list[ForwardModelStep]:
        return create_list_of_forward_model_steps_to_run(
            installed_steps,
            substitutions,
            config_dict,
            cls.PREINSTALLED_FORWARD_MODEL_STEPS,
            cls.ENV_PR_FM_STEP,
        )

    @classmethod
    def _read_summary_keys(cls, config_dict) -> list[str]:
        return [
            item
            for sublist in config_dict.get(ConfigKeys.SUMMARY, [])
            for item in sublist
        ]

    @classmethod
    def _log_config_file(cls, config_file: str, config_file_contents: str) -> None:
        """
        Logs what configuration was used to start ert. Because the config
        parsing is quite convoluted we are not able to remove all the comments,
        but the easy ones are filtered out.
        """
        config_context = ""
        for line in config_file_contents.split("\n"):
            line = line.strip()
            if not line or line.startswith("--"):
                continue
            if "--" in line and not any(x in line for x in ['"', "'"]):
                # There might be a comment in this line, but it could
                # also be an argument to a job, so we do a quick check
                line = line.split("--")[0].rstrip()
            if any(
                kw in line
                for kw in [
                    "FORWARD_MODEL",
                    "LOAD_WORKFLOW",
                    "LOAD_WORKFLOW_JOB",
                    "HOOK_WORKFLOW",
                    "WORKFLOW_JOB_DIRECTORY",
                ]
            ):
                continue
            config_context += line + "\n"
        logger.info(
            f"Content of the configuration file ({config_file}):\n{config_context}"
        )

    @classmethod
    def _log_config_dict(cls, content_dict: dict[str, Any]) -> None:
        # The content of the message is sanitized before beeing sendt to App Insights
        # to make sure GDPR-rules are not violated. In doing do, the message length
        # will typically increase a bit. To Avoid hiting the App Insights' hard limit
        # of message length, the limit is set to 80% of
        # MAX_MESSAGE_LENGTH_APP_INSIGHTS = 32768
        SAFE_MESSAGE_LENGTH_LIMIT = 26214  # <= MAX_MESSAGE_LENGTH_APP_INSIGHTS * 0.8
        try:
            config_dict_content = pprint.pformat(content_dict)
        except Exception as err:
            config_dict_content = str(content_dict)
            logger.warning(
                "Logging of config dict could not be formatted for "
                f"enhanced readability. {err}"
            )
        config_dict_content_length = len(config_dict_content)
        if config_dict_content_length > SAFE_MESSAGE_LENGTH_LIMIT:
            config_sections = _split_string_into_sections(
                config_dict_content, SAFE_MESSAGE_LENGTH_LIMIT
            )
            section_count = len(config_sections)
            for i, section in enumerate(config_sections):
                logger.info(
                    "Content of the config_dict "
                    f"(part {i + 1}/{section_count}): {section}"
                )
        else:
            logger.info(f"Content of the config_dict: {config_dict_content}")

    @cached_property
    def ensemble_size(self) -> int:
        config_num_realizations = self.runpath_config.num_realizations
        if (
            self.analysis_config.design_matrix is not None
            and (
                dm_active_realizations
                := self.analysis_config.design_matrix.active_realizations
            )
            is not None
        ) and (
            dm_num_realizations := len(dm_active_realizations)
        ) != config_num_realizations:
            msg = (
                f"NUM_REALIZATIONS ({config_num_realizations}) is "
                + (
                    "greater "
                    if dm_num_realizations < config_num_realizations
                    else "less "
                )
                + f"than the number of realizations in DESIGN_MATRIX "
                f"({dm_num_realizations}). Using the realizations from "
                + (
                    f"DESIGN_MATRIX ({dm_num_realizations})"
                    if dm_num_realizations < config_num_realizations
                    else f"NUM_REALIZATIONS ({config_num_realizations})"
                )
            )
            ConfigWarning.warn(msg)
            return min(config_num_realizations, dm_num_realizations)
        return config_num_realizations

    @cached_property
    def active_realizations(self) -> list[bool]:
        if (
            self.analysis_config.design_matrix is not None
            and (
                dm_active_realizations
                := self.analysis_config.design_matrix.active_realizations
            )
            is not None
        ):
            return dm_active_realizations[: self.ensemble_size]
        return [True for _ in range(self.ensemble_size)]

    @classmethod
    def _log_custom_forward_model_steps(cls, user_config: ConfigDict) -> None:
        for fm_step, (fm_step_filename, _) in user_config.get(
            ConfigKeys.INSTALL_JOB, []
        ):
            fm_configuration = EMPTY_LINES.sub(
                "\n", (Path(fm_step_filename).read_text(encoding="utf-8").strip())
            )
            logger.info(
                f"Custom forward_model_step {fm_step} installed as: {fm_configuration}"
            )

    @staticmethod
    def apply_config_content_defaults(content_dict: dict, config_dir: str):
        if ConfigKeys.ENSPATH not in content_dict:
            content_dict[ConfigKeys.ENSPATH] = path.join(
                config_dir, ErtConfig.DEFAULT_ENSPATH
            )
        if ConfigKeys.RUNPATH_FILE not in content_dict:
            content_dict[ConfigKeys.RUNPATH_FILE] = path.join(
                config_dir, ErtConfig.DEFAULT_RUNPATH_FILE
            )
        elif not path.isabs(content_dict[ConfigKeys.RUNPATH_FILE]):
            content_dict[ConfigKeys.RUNPATH_FILE] = path.normpath(
                path.join(config_dir, content_dict[ConfigKeys.RUNPATH_FILE])
            )

    @classmethod
    def read_site_config(cls) -> ConfigDict:
        site_config_file = site_config_location()
        if not site_config_file:
            return ConfigDict()
        return parse_contents(
            read_file(site_config_file),
            file_name=site_config_file,
            schema=init_site_config_schema(),
        )

    @classmethod
    def _read_user_config_contents(cls, user_config: str, file_name: str) -> ConfigDict:
        return parse_contents(
            user_config, file_name=file_name, schema=init_user_config_schema()
        )

    @classmethod
    def _merge_user_and_site_config(
        cls, user_config_dict: ConfigDict, site_config_dict: ConfigDict
    ) -> ConfigDict:
        for keyword, value in site_config_dict.items():
            if keyword == "QUEUE_OPTION":
                filtered_queue_options = []
                for queue_option in value:
                    option_name = queue_option[1]
                    if option_name in user_config_dict:
                        continue
                    filtered_queue_options.append(queue_option)
                user_config_dict["QUEUE_OPTION"] = (
                    filtered_queue_options + user_config_dict.get("QUEUE_OPTION", [])
                )
            elif isinstance(value, list):
                original_entries: list = user_config_dict.get(keyword, [])
                user_config_dict[keyword] = value + original_entries
            elif keyword not in user_config_dict:
                user_config_dict[keyword] = value
        return user_config_dict

    @classmethod
    def _read_user_config_and_apply_site_config(
        cls,
        user_config_contents: str,
        user_config_file: str,
        site_config_dict: ConfigDict,
    ) -> ConfigDict:
        user_config_dict = cls._read_user_config_contents(
            user_config_contents,
            file_name=user_config_file,
        )
        cls._log_custom_forward_model_steps(user_config_dict)
        return cls._merge_user_and_site_config(user_config_dict, site_config_dict)

    @classmethod
    def _validate_dict(
        cls, config_dict, config_file: str
    ) -> list[ErrorInfo | ConfigValidationError]:
        errors = []

        if ConfigKeys.SUMMARY in config_dict and ConfigKeys.ECLBASE not in config_dict:
            errors.append(
                ErrorInfo(
                    message="When using SUMMARY keyword, "
                    "the config must also specify ECLBASE",
                    filename=config_file,
                ).set_context_keyword(config_dict[ConfigKeys.SUMMARY][0][0])
            )
        return errors

    def forward_model_step_name_list(self) -> list[str]:
        return [j.name for j in self.forward_model_steps]

    @property
    def env_pr_fm_step(self) -> dict[str, dict[str, Any]]:
        return self.ENV_PR_FM_STEP

    @staticmethod
    def _create_observations(
        obs_config_content: dict[str, HistoryValues | SummaryValues | GenObsValues]
        | None,
        ensemble_config: EnsembleConfig,
        time_map: list[datetime] | None,
        history: HistorySource,
    ) -> EnkfObs:
        if not obs_config_content:
            return EnkfObs({}, [])
        obs_vectors: dict[str, ObsVector] = {}
        obs_time_list: Sequence[datetime] = []
        if ensemble_config.refcase is not None:
            obs_time_list = ensemble_config.refcase.all_dates
        elif time_map is not None:
            obs_time_list = time_map

        time_len = len(obs_time_list)
        config_errors: list[ErrorInfo] = []
        for obs_name, values in obs_config_content:
            try:
                if type(values) is HistoryValues:
                    obs_vectors.update(
                        **EnkfObs._handle_history_observation(
                            ensemble_config,
                            values,
                            obs_name,
                            history,
                            time_len,
                        )
                    )
                elif type(values) is SummaryValues:
                    obs_vectors.update(
                        **EnkfObs._handle_summary_observation(
                            values,
                            obs_name,
                            obs_time_list,
                            bool(ensemble_config.refcase),
                        )
                    )
                elif type(values) is GenObsValues:
                    obs_vectors.update(
                        **EnkfObs._handle_general_observation(
                            ensemble_config,
                            values,
                            obs_name,
                            obs_time_list,
                            bool(ensemble_config.refcase),
                        )
                    )
                else:
                    config_errors.append(
                        ErrorInfo(
                            message=(
                                f"Unknown ObservationType {type(values)} for {obs_name}"
                            )
                        ).set_context(obs_name)
                    )
                    continue
            except ObservationConfigError as err:
                config_errors.extend(err.errors)
            except ValueError as err:
                config_errors.append(ErrorInfo(message=str(err)).set_context(obs_name))

        if config_errors:
            raise ObservationConfigError.from_collected(config_errors)

        return EnkfObs(obs_vectors, obs_time_list)


def _split_string_into_sections(string: str, section_length: int) -> list[str]:
    """
    Splits a string into sections of length section_length and returns it as a list.

    If section_length is set to 0 or less, no sectioning is performed and the entire
    input string is returned as one section in a list
    """
    if section_length < 1:
        return [string]
    return [
        string[i : i + section_length] for i in range(0, len(string), section_length)
    ]


def _get_files_in_directory(job_path, errors):
    if not path.isdir(job_path):
        errors.append(
            ConfigValidationError.with_context(
                f"Unable to locate job directory {job_path!r}", job_path
            )
        )
        return []
    files = list(
        filter(
            path.isfile,
            (path.abspath(path.join(job_path, f)) for f in os.listdir(job_path)),
        )
    )

    if files == []:
        ConfigWarning.warn(f"No files found in job directory {job_path}", job_path)
    return files


def _substitutions_from_dict(config_dict) -> dict[str, str]:
    substitutions = {}

    for key, val in config_dict.get("DEFINE", []):
        substitutions[key] = val

    if "<CONFIG_PATH>" not in substitutions:
        substitutions["<CONFIG_PATH>"] = os.getcwd()

    for key, val in config_dict.get("DATA_KW", []):
        substitutions[key] = val

    return substitutions


def uppercase_subkeys_and_stringify_subvalues(
    nested_dict: dict[str, dict[str, Any]],
) -> dict[str, dict[str, str]]:
    fixed_dict: dict[str, dict[str, str]] = {}
    for key, value in nested_dict.items():
        fixed_dict[key] = {
            subkey.upper(): str(subvalue) for subkey, subvalue in value.items()
        }
    return fixed_dict


@no_type_check
def _forward_model_step_from_config_contents(
    config_contents: str, config_file: str, name: str | None = None
) -> "ForwardModelStep":
    if name is None:
        name = os.path.basename(config_file)

    schema = init_forward_model_schema()

    content_dict = parse_contents(
        config_contents, file_name=config_file, schema=schema, pre_defines=[]
    )

    specified_arg_types: list[tuple[int, str]] = content_dict.get(
        ForwardModelStepKeys.ARG_TYPE, []
    )

    specified_max_args: int = content_dict.get("MAX_ARG", 0)
    specified_min_args: int = content_dict.get("MIN_ARG", 0)

    arg_types_list = parse_arg_types_list(
        specified_arg_types, specified_min_args, specified_max_args
    )

    environment = {k: v for [k, v] in content_dict.get("ENV", [])}
    default_mapping = {k: v for [k, v] in content_dict.get("DEFAULT", [])}

    return ForwardModelStep(
        name=name,
        executable=content_dict.get("EXECUTABLE"),
        stdin_file=content_dict.get("STDIN"),
        stdout_file=content_dict.get("STDOUT"),
        stderr_file=content_dict.get("STDERR"),
        start_file=content_dict.get("START_FILE"),
        target_file=content_dict.get("TARGET_FILE"),
        error_file=content_dict.get("ERROR_FILE"),
        max_running_minutes=content_dict.get("MAX_RUNNING_MINUTES"),
        min_arg=content_dict.get("MIN_ARG"),
        max_arg=content_dict.get("MAX_ARG"),
        arglist=content_dict.get("ARGLIST", []),
        arg_types=arg_types_list,
        environment=environment,
        required_keywords=content_dict.get("REQUIRED", []),
        default_mapping=default_mapping,
    )


# Due to circular dependency in type annotations between
# ErtConfig -> WorkflowJob -> ErtScript -> ErtConfig
ErtConfig.model_rebuild()
