from __future__ import annotations

import copy
import logging
import os
import pprint
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import datetime
from functools import cached_property
from os import path
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast, overload

import polars as pl
from numpy.random import SeedSequence
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from pydantic import ValidationError as PydanticValidationError

from ert.substitutions import Substitutions

from ._create_observation_dataframes import create_observation_dataframes
from ._design_matrix_validator import DesignMatrixValidator
from ._observations import (
    HistoryObservation,
    Observation,
    SummaryObservation,
    make_observations,
)
from .analysis_config import AnalysisConfig
from .ensemble_config import EnsembleConfig
from .forward_model_step import (
    ForwardModelJSON,
    ForwardModelStep,
    ForwardModelStepJSON,
    ForwardModelStepValidationError,
    ForwardModelStepWarning,
    SiteInstalledForwardModelStep,
    SiteOrUserForwardModelStep,
    UserInstalledForwardModelStep,
)
from .gen_data_config import GenDataConfig
from .gen_kw_config import DataSource, GenKwConfig
from .model_config import ModelConfig
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
    ObservationConfigError,
    init_forward_model_schema,
    init_user_config_schema,
    parse_contents,
    read_file,
)
from .parsing.observations_parser import ObservationDict
from .queue_config import KnownQueueOptions, QueueConfig
from .refcase import Refcase
from .workflow import Workflow
from .workflow_fixtures import fixtures_per_hook
from .workflow_job import (
    BaseErtScriptWorkflow,
    ErtScriptLoadFailure,
    ErtScriptWorkflow,
    WorkflowJob,
    workflow_job_from_file,
)

if TYPE_CHECKING:
    from ert.plugins import ErtRuntimePlugins

    from .parameter_config import ParameterConfig

logger = logging.getLogger(__name__)

EMPTY_LINES = re.compile(r"\n[\s\n]*\n")

ECL_BASE_DEPRECATION_MSG = (
    "Substitution template <ECL_BASE> is deprecated and "
    "will be removed in the future. Please use <ECLBASE> instead."
)


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
    real_iter_substituter = context_substitutions.real_iter_substituter(iens, itr)

    class Substituter:
        def __init__(self, fm_step: ForwardModelStep) -> None:
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
                    key: real_iter_substituter.substitute(val)
                    for key, val in fm_step.private_args.items()
                }
            )

        @overload
        def substitute(self, string: str) -> str: ...

        @overload
        def substitute(self, string: None) -> None: ...

        def substitute(self, string: str | None) -> str | None:
            if string is None:
                return string
            string = self.copy_private_args.substitute(
                string, self.substitution_context_hint, 1, warn_max_iter=False
            )
            return real_iter_substituter.substitute(string)

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
        Path(file_path).read_text(encoding="utf-8")
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


def read_templates(config_dict: ConfigDict) -> list[tuple[str, str]]:
    templates: list[tuple[str, str]] = []
    if ConfigKeys.DATA_FILE in config_dict and ConfigKeys.ECLBASE in config_dict:
        source_file = config_dict[ConfigKeys.DATA_FILE]
        target_file = config_dict[ConfigKeys.ECLBASE].replace("%d", "<IENS>") + ".DATA"
        check_non_utf_chars(source_file)
        templates.append((source_file, target_file))

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
    site_installed_workflows_jobs: dict[str, WorkflowJob] | None = None,
) -> dict[str, WorkflowJob]:
    user_installed_workflow_job_info = content_dict.get(
        ConfigKeys.LOAD_WORKFLOW_JOB, []
    )
    user_installed_workflow_job_dir_info = content_dict.get(
        ConfigKeys.WORKFLOW_JOB_DIRECTORY, []
    )

    workflow_jobs = (
        copy.copy(site_installed_workflows_jobs)
        if site_installed_workflows_jobs
        else {}
    )

    errors: list[ErrorInfo | ConfigValidationError] = []

    for user_workflow_job in user_installed_workflow_job_info:
        try:
            # workflow_job_from_file only throws error if a
            # non-readable file is provided.
            # Non-existing files are caught by the new parser
            user_job = workflow_job_from_file(
                config_file=user_workflow_job[0],
                name=None if len(user_workflow_job) == 1 else user_workflow_job[1],
                origin="user",
            )
            name = user_job.name
            if name in workflow_jobs:
                ConfigWarning.warn(
                    f"Duplicate workflow jobs with name {name!r}, choosing "
                    f"{user_job.location()!r} over "
                    f"{workflow_jobs[name].location()!r}",
                    name,
                )
            workflow_jobs[name] = user_job
        except ErtScriptLoadFailure as err:
            ConfigWarning.warn(
                f"Loading workflow job {user_workflow_job[0]!r}"
                f" failed with '{err}'. It will not be loaded.",
                user_workflow_job[0],
            )
        except ConfigValidationError as err:
            errors.append(ErrorInfo(message=str(err)).set_context(user_workflow_job[0]))

    for user_job_path in user_installed_workflow_job_dir_info:
        for user_job_file in _get_files_in_directory(user_job_path, errors):
            try:
                user_job = workflow_job_from_file(
                    config_file=user_job_file, origin="user"
                )
                name = user_job.name
                if name in workflow_jobs:
                    ConfigWarning.warn(
                        f"Duplicate workflow jobs with name {name!r}, choosing "
                        f"{user_job.location()!r} over "
                        f"{workflow_jobs[name].location()!r}",
                        name,
                    )
                workflow_jobs[name] = user_job
            except ErtScriptLoadFailure as err:
                ConfigWarning.warn(
                    f"Loading workflow job {user_job_file!r}"
                    f" failed with '{err}'. It will not be loaded.",
                    user_job_file,
                )
            except ConfigValidationError as err:
                errors.append(ErrorInfo(message=str(err)).set_context(user_job_path))
    if errors:
        raise ConfigValidationError.from_collected(errors)

    return workflow_jobs


def create_and_hook_workflows(
    hook_workflow_info: list[tuple[str, HookRuntime]],
    workflow_info: list[tuple[str, str]],
    workflow_jobs: dict[str, WorkflowJob],
    substitutions: dict[str, str],
) -> tuple[dict[str, Workflow], defaultdict[HookRuntime, list[Workflow]]]:
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
                        job.load_ert_script_class().validate(args)
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
            if not isinstance(job, BaseErtScriptWorkflow):
                continue

            ert_script_class = job.load_ert_script_class()
            ert_script_instance = ert_script_class()
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


def workflows_from_dict(
    content_dict: ConfigDict,
    substitutions: dict[str, str],
    site_installed_workflows_jobs: Mapping[str, WorkflowJob] | None = None,
) -> tuple[
    dict[str, WorkflowJob],
    dict[str, Workflow],
    defaultdict[HookRuntime, list[Workflow]],
]:
    workflow_jobs = workflow_jobs_from_dict(
        content_dict,
        dict(copy.copy(site_installed_workflows_jobs))
        if site_installed_workflows_jobs
        else {},
    )
    workflows, hooked_workflows = create_and_hook_workflows(
        content_dict.get(ConfigKeys.HOOK_WORKFLOW, []),
        content_dict.get(ConfigKeys.LOAD_WORKFLOW, []),
        workflow_jobs,
        substitutions,
    )
    return workflow_jobs, workflows, hooked_workflows


def installed_forward_model_steps_from_dict(
    config_dict: ConfigDict,
) -> dict[str, UserInstalledForwardModelStep]:
    errors: list[ErrorInfo | ConfigValidationError] = []
    fm_steps: dict[str, UserInstalledForwardModelStep] = {}
    for name, (fm_step_config_file, config_contents) in config_dict.get(
        ConfigKeys.INSTALL_JOB, []
    ):
        fm_step_config_file = path.abspath(fm_step_config_file)
        try:
            new_fm_step = forward_model_step_from_config_contents(
                config_contents, name=name, config_file=fm_step_config_file
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
                new_fm_step = forward_model_step_from_config_contents(
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
    config_dict: ConfigDict,
    preinstalled_forward_model_steps: Mapping[str, ForwardModelStep],
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
                    **(fm_json_for_validation["environment"] or {}),
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

            except Exception as e:
                ConfigWarning.warn(
                    f"Unexpected plugin forward model exception: {e!s}",
                    context=fm_step.name,
                )
    dm_validator.validate_design_matrix_merge()

    if errors:
        raise ConfigValidationError.from_collected(errors)

    return fm_steps


def log_observation_keys(
    observations: list[ObservationDict],
) -> None:
    observation_type_counts = Counter(o["type"].value for o in observations)
    observation_keyword_counts = Counter(
        "SEGMENT" if key == "segments" else str(key)
        for o in observations
        for key in o
        if key not in {"name", "type"}
    )

    if "HISTORY_OBSERVATION" in observation_type_counts:
        msg = (
            "HISTORY_OBSERVATION is deprecated and will be removed. "
            "Please use SUMMARY_OBSERVATION instead."
        )
        ConfigWarning.warn(msg)
        logger.warning(msg)

    logger.info(
        f"Count of observation types:\n\t{dict(observation_type_counts)}\n"
        f"Count of observation keywords:\n\t{dict(observation_keyword_counts)}"
    )


RESERVED_KEYWORDS = ["realization", "IENS", "ITER"]

USER_CONFIG_SCHEMA = init_user_config_schema()


class ErtConfig(BaseModel):
    DEFAULT_ENSPATH: ClassVar[str] = "storage"
    DEFAULT_RUNPATH_FILE: ClassVar[str] = ".ert_runpath_list"
    PREINSTALLED_FORWARD_MODEL_STEPS: ClassVar[Mapping[str, ForwardModelStep]] = {}
    PREINSTALLED_WORKFLOWS: ClassVar[dict[str, WorkflowJob]] = {}
    ENV_PR_FM_STEP: ClassVar[dict[str, dict[str, Any]]] = {}
    ENV_VARIABLES: ClassVar[dict[str, str]] = {}
    QUEUE_OPTIONS: ClassVar[KnownQueueOptions | None] = None
    RESERVED_KEYWORDS: ClassVar[list[str]] = RESERVED_KEYWORDS
    ENV_VARS: ClassVar[dict[str, str]] = {}

    substitutions: dict[str, str] = Field(default_factory=dict)
    ensemble_config: EnsembleConfig = Field(default_factory=EnsembleConfig)
    ens_path: str = DEFAULT_ENSPATH
    env_vars: dict[str, str] = Field(default_factory=dict)
    random_seed: int = Field(default_factory=lambda: _seed_sequence(None))
    analysis_config: AnalysisConfig = Field(default_factory=AnalysisConfig)
    queue_config: QueueConfig = Field(default_factory=QueueConfig)
    workflow_jobs: dict[str, WorkflowJob] = Field(default_factory=dict)
    workflows: dict[str, Workflow] = Field(default_factory=dict)
    hooked_workflows: defaultdict[HookRuntime, list[Workflow]] = Field(
        default_factory=lambda: defaultdict(lambda: cast(list[Workflow], []))
    )
    runpath_file: Path = Path(DEFAULT_RUNPATH_FILE)

    ert_templates: list[tuple[str, str]] = Field(default_factory=list)

    forward_model_steps: list[SiteOrUserForwardModelStep] = Field(default_factory=list)
    runpath_config: ModelConfig = Field(default_factory=ModelConfig)
    user_config_file: str = "no_config"
    config_path: str = Field(init=False, default="")
    observation_declarations: list[Observation] = Field(default_factory=list)
    time_map: list[datetime] | None = None
    history_source: HistorySource = HistorySource.REFCASE_HISTORY
    refcase: Refcase | None = None
    _observations: dict[str, pl.DataFrame] | None = PrivateAttr(None)

    @property
    def observations(self) -> dict[str, pl.DataFrame]:
        if self._observations is None:
            computed = create_observation_dataframes(
                self.observation_declarations,
                self.refcase,
                cast(
                    GenDataConfig | None,
                    self.ensemble_config.response_configs.get("gen_data", None),
                ),
                self.time_map,
                self.history_source,
            )
            self._observations = computed
            return computed
        return self._observations

    @model_validator(mode="after")
    def set_fields(self) -> Self:
        self.config_path = (
            path.dirname(path.abspath(self.user_config_file))
            if self.user_config_file
            else os.getcwd()
        )
        return self

    @model_validator(mode="after")
    def validate_genkw_parameter_name_overlap(self) -> Self:
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
    def validate_dm_parameter_name_overlap(self) -> Self:
        if not self.analysis_config.design_matrix:
            return self
        dm_param_configs = self.analysis_config.design_matrix.parameter_configurations
        overlapping_parameter_names = [
            parameter_definition.name
            for parameter_definition in dm_param_configs
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

    @model_validator(mode="after")
    def log_ensemble_config_contents(self) -> Self:
        all_parameters = self.parameter_configurations_with_design_matrix
        parameter_type_count = Counter(parameter.type for parameter in all_parameters)
        logger.info(
            f"EnsembleConfig contains parameters of type {dict(parameter_type_count)}"
        )
        return self

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
    def with_plugins(runtime_plugins: ErtRuntimePlugins) -> type[ErtConfig]:
        class ErtConfigWithPlugins(ErtConfig):
            PREINSTALLED_FORWARD_MODEL_STEPS: ClassVar[
                Mapping[str, SiteInstalledForwardModelStep]
            ] = runtime_plugins.installed_forward_model_steps
            PREINSTALLED_WORKFLOWS = dict(runtime_plugins.installed_workflow_jobs)
            ENV_PR_FM_STEP: ClassVar[dict[str, dict[str, Any]]] = (
                uppercase_subkeys_and_stringify_subvalues(
                    {k: dict(v) for k, v in runtime_plugins.env_pr_fm_step.items()}
                )
            )
            ENV_VARS = dict(runtime_plugins.environment_variables)
            QUEUE_OPTIONS = runtime_plugins.queue_options

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
        user_config_dict = cls._config_dict_from_contents(
            user_config_contents,
            user_config_file,
        )
        cls._log_config_dict(user_config_dict)
        return cls.from_dict(user_config_dict)

    @classmethod
    def _config_dict_from_contents(
        cls,
        user_config_contents: str,
        config_file_name: str,
    ) -> ConfigDict:
        user_config_dict = cls._read_user_config_contents(
            user_config_contents,
            file_name=config_file_name,
        )
        cls._log_custom_forward_model_steps(user_config_dict)

        config_dir = path.abspath(path.dirname(config_file_name))
        cls.apply_config_content_defaults(user_config_dict, config_dir)
        return user_config_dict

    @classmethod
    def from_file_contents(
        cls,
        user_config_contents: str,
        config_file_name: str = "./config.ert",
    ) -> Self:
        return cls.from_dict(
            cls._config_dict_from_contents(
                user_config_contents,
                config_file_name,
            )
        )

    @classmethod
    def from_dict(cls, config_dict: ConfigDict) -> Self:
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

        workflow_jobs: dict[str, WorkflowJob] = {}
        workflows: dict[str, Workflow] = {}
        hooked_workflows: dict[HookRuntime, list[Workflow]] = {}
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
            site_installed_forward_model_steps = dict(
                copy.deepcopy(cls.PREINSTALLED_FORWARD_MODEL_STEPS)
            )

            user_installed_forward_model_steps = (
                installed_forward_model_steps_from_dict(config_dict)
            )

            overwritten_fm_steps = set(site_installed_forward_model_steps).intersection(
                user_installed_forward_model_steps
            )
            if overwritten_fm_steps:
                msg = (
                    f"The following forward model steps from site configurations "
                    f"have been overwritten by user: {sorted(overwritten_fm_steps)}"
                )
                logger.warning(msg)

            installed_forward_model_steps = (
                site_installed_forward_model_steps | user_installed_forward_model_steps
            )

        except ConfigValidationError as e:
            errors.append(e)

        try:
            queue_config = QueueConfig.from_dict(
                config_dict, site_queue_options=cls.QUEUE_OPTIONS
            )

            substitutions["<NUM_CPU>"] = str(queue_config.queue_options.num_cpu)

        except ConfigValidationError as err:
            errors.append(err)

        obs_config_args = config_dict.get(ConfigKeys.OBS_CONFIG)
        obs_configs: list[Observation] = []
        try:
            if obs_config_args:
                obs_config_file, obs_config_input = obs_config_args
                log_observation_keys(obs_config_input)
                obs_configs = make_observations(
                    os.path.dirname(obs_config_file),
                    obs_config_input,
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
                    obs.key
                    for obs in obs_configs
                    if isinstance(obs, HistoryObservation | SummaryObservation)
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
            dm_errors: list[ErrorInfo | ConfigValidationError] = []
            dm_params = {x.name for x in dm.parameter_configurations}
            overwrite_params = [
                cfg.name
                for cfg in ensemble_config.parameter_configs.values()
                if isinstance(cfg, GenKwConfig) and cfg.name in dm_params
            ]
            if overwrite_params:
                param_sampled = [
                    k
                    for k in overwrite_params
                    if analysis_config.design_matrix.parameter_priority[k]
                    == DataSource.SAMPLED
                ]
                param_design = [
                    k
                    for k in overwrite_params
                    if analysis_config.design_matrix.parameter_priority[k]
                    == DataSource.DESIGN_MATRIX
                ]
                if param_sampled:
                    ConfigWarning.warn(
                        f"Parameters {param_sampled} "
                        "are also defined in design matrix, but due to the sampled"
                        " priority they will remain as such."
                    )
                if param_design:
                    ConfigWarning.warn(
                        f"Parameters {param_design} "
                        "will be overridden by design matrix. This will cause "
                        "updates to be turned off for these parameters."
                    )

            if dm_errors:
                raise ConfigValidationError.from_collected(dm_errors)

        env_vars = {}
        substituter = Substitutions(substitutions)
        history_source = config_dict.get(
            ConfigKeys.HISTORY_SOURCE, HistorySource.REFCASE_HISTORY
        )

        # Insert env vars from plugins/site config
        for key, val in cls.ENV_VARS.items():
            env_vars[key] = substituter.substitute(val)

        user_configured_ = set()
        for key, val in config_dict.get("SETENV", []):
            if key in user_configured_:
                logger.warning(
                    f"User configured environment variable {key} re-written by user: "
                    f"{env_vars[key]}->{val}"
                )
            elif key in cls.ENV_VARS:
                logger.warning(
                    f"Site configured environment variable {key} re-written by user: "
                    f"{env_vars[key]}->{val}"
                )

            user_configured_.add(key)
            env_vars[key] = substituter.substitute(val)

        try:
            refcase = Refcase.from_config_dict(config_dict)
            cls_config = cls(
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
                forward_model_steps=cls._create_list_of_forward_model_steps_to_run(
                    installed_forward_model_steps,
                    substitutions,
                    config_dict,
                ),
                runpath_config=model_config,
                user_config_file=config_file_path,
                observation_declarations=list(obs_configs),
                time_map=time_map,
                history_source=history_source,
                refcase=refcase,
            )

            # The observations are created here because create_observation_dataframes
            # will perform additonal validation which needs the context in
            # obs_configs which is stripped by pydantic
            cls_config._observations = create_observation_dataframes(
                obs_configs,
                refcase,
                cast(
                    GenDataConfig | None,
                    ensemble_config.response_configs.get("gen_data", None),
                ),
                time_map,
                history_source,
            )
        except PydanticValidationError as err:
            raise ConfigValidationError.from_pydantic(err) from err
        return cls_config

    @classmethod
    def _create_list_of_forward_model_steps_to_run(
        cls,
        installed_steps: dict[str, ForwardModelStep],
        substitutions: dict[str, str],
        config_dict: ConfigDict,
    ) -> list[ForwardModelStep]:
        return create_list_of_forward_model_steps_to_run(
            installed_steps,
            substitutions,
            config_dict,
            cls.PREINSTALLED_FORWARD_MODEL_STEPS,
            cls.ENV_PR_FM_STEP,
        )

    @classmethod
    def _read_summary_keys(cls, config_dict: ConfigDict) -> list[str]:
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

    @property
    def parameter_configurations_with_design_matrix(self) -> list[ParameterConfig]:
        if self.analysis_config.design_matrix is not None:
            return self.analysis_config.design_matrix.merge_with_existing_parameters(
                self.ensemble_config.parameter_configuration
            )
        return self.ensemble_config.parameter_configuration

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
    def apply_config_content_defaults(
        content_dict: ConfigDict, config_dir: str
    ) -> None:
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
    def _read_user_config_contents(cls, user_config: str, file_name: str) -> ConfigDict:
        return parse_contents(
            user_config, file_name=file_name, schema=USER_CONFIG_SCHEMA
        )

    @classmethod
    def _validate_dict(
        cls, config_dict: ConfigDict, config_file: str
    ) -> list[ErrorInfo | ConfigValidationError]:
        errors: list[ErrorInfo | ConfigValidationError] = []

        if ConfigKeys.SUMMARY in config_dict and ConfigKeys.ECLBASE not in config_dict:
            errors.append(
                ErrorInfo(
                    message="When using SUMMARY keyword, "
                    "the config must also specify ECLBASE",
                    filename=config_file,
                ).set_context(config_dict[ConfigKeys.SUMMARY][0])
            )
        return errors

    def forward_model_step_name_list(self) -> list[str]:
        return [j.name for j in self.forward_model_steps]

    @property
    def env_pr_fm_step(self) -> dict[str, dict[str, Any]]:
        return self.ENV_PR_FM_STEP


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


def _get_files_in_directory(
    job_path: str, errors: list[ErrorInfo | ConfigValidationError]
) -> list[str]:
    if not path.isdir(job_path):
        errors.append(
            ConfigValidationError(
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


def _substitutions_from_dict(config_dict: ConfigDict) -> dict[str, str]:
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


def forward_model_step_from_config_contents(
    config_contents: str,
    config_file: str,
    name: str | None = None,
) -> UserInstalledForwardModelStep:
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

    return UserInstalledForwardModelStep(
        name=name,
        executable=content_dict["EXECUTABLE"],
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
