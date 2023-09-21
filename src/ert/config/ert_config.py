# mypy: ignore-errors
import copy
import logging
import os
import pkgutil
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from os.path import dirname
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
    overload,
)

from typing_extensions import Self

from ert.substitution_list import SubstitutionList

from .analysis_config import AnalysisConfig
from .ensemble_config import EnsembleConfig
from .ext_job import ExtJob
from .hook_runtime import HookRuntime
from .model_config import ModelConfig
from .parsing import (
    ConfigDict,
    ConfigKeys,
    ConfigValidationError,
    ConfigWarning,
    ErrorInfo,
    init_site_config_schema,
    init_user_config_schema,
    lark_parse,
)
from .queue_config import QueueConfig
from .workflow import Workflow
from .workflow_job import ErtScriptLoadFailure, WorkflowJob

if TYPE_CHECKING:
    from importlib.abc import FileLoader


logger = logging.getLogger(__name__)


def site_config_location() -> str:
    if "ERT_SITE_CONFIG" in os.environ:
        return os.environ["ERT_SITE_CONFIG"]
    ert_shared_loader = cast("FileLoader", pkgutil.get_loader("ert.shared"))
    return dirname(ert_shared_loader.get_filename()) + "/share/ert/site-config"


@dataclass
class ErtConfig:  # pylint: disable=too-many-instance-attributes
    DEFAULT_ENSPATH: ClassVar[str] = "storage"
    DEFAULT_RUNPATH_FILE: ClassVar[str] = ".ert_runpath_list"

    substitution_list: SubstitutionList = field(default_factory=SubstitutionList)
    ensemble_config: EnsembleConfig = field(default_factory=EnsembleConfig)
    ens_path: str = DEFAULT_ENSPATH
    env_vars: Dict[str, str] = field(default_factory=dict)
    random_seed: Optional[str] = None
    analysis_config: AnalysisConfig = field(default_factory=AnalysisConfig)
    queue_config: QueueConfig = field(default_factory=QueueConfig)
    workflow_jobs: Dict[str, WorkflowJob] = field(default_factory=dict)
    workflows: Dict[str, Workflow] = field(default_factory=dict)
    hooked_workflows: Dict[HookRuntime, List[Workflow]] = field(default_factory=dict)
    runpath_file: Path = Path(DEFAULT_RUNPATH_FILE)
    ert_templates: List[List[str]] = field(default_factory=list)
    installed_jobs: Dict[str, ExtJob] = field(default_factory=dict)
    forward_model_list: List[ExtJob] = field(default_factory=list)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    user_config_file: str = "no_config"
    config_path: str = field(init=False)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ErtConfig):
            return False

        return all(getattr(self, attr) == getattr(other, attr) for attr in vars(self))

    def __post_init__(self) -> None:
        self.config_path = (
            os.path.dirname(os.path.abspath(self.user_config_file))
            if self.user_config_file
            else os.getcwd()
        )

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
        user_config_dict = ErtConfig.read_user_config(user_config_file)
        config_dir = os.path.abspath(os.path.dirname(user_config_file))
        ErtConfig._log_config_file(user_config_file)
        ErtConfig._log_config_dict(user_config_dict)
        ErtConfig.apply_config_content_defaults(user_config_dict, config_dir)
        return ErtConfig.from_dict(user_config_dict)

    @classmethod
    def from_dict(cls, config_dict) -> Self:
        substitution_list = SubstitutionList.from_dict(config_dict=config_dict)
        runpath_file = config_dict.get(
            ConfigKeys.RUNPATH_FILE, ErtConfig.DEFAULT_RUNPATH_FILE
        )
        substitution_list["<RUNPATH_FILE>"] = runpath_file
        config_dir = substitution_list.get("<CONFIG_PATH>", "")
        config_file = substitution_list.get("<CONFIG_FILE>", "no_config")
        config_file_path = os.path.join(config_dir, config_file)

        errors = cls._validate_dict(config_dict, config_file)

        if errors:
            raise ConfigValidationError.from_collected(errors)

        try:
            ensemble_config = EnsembleConfig.from_dict(config_dict=config_dict)
        except ConfigValidationError as err:
            errors.append(err)

        workflow_jobs = {}
        workflows = {}
        hooked_workflows = {}
        installed_jobs = {}
        model_config = None

        try:
            model_config = ModelConfig.from_dict(config_dict)
            runpath = model_config.runpath_format_string
            eclbase = model_config.eclbase_format_string
            substitution_list["<RUNPATH>"] = runpath
            substitution_list["<ECL_BASE>"] = eclbase
            substitution_list["<ECLBASE>"] = eclbase
        except ConfigValidationError as e:
            errors.append(e)

        try:
            workflow_jobs, workflows, hooked_workflows = cls._workflows_from_dict(
                config_dict, substitution_list
            )
        except ConfigValidationError as e:
            errors.append(e)

        try:
            installed_jobs = cls._installed_jobs_from_dict(config_dict)
        except ConfigValidationError as e:
            errors.append(e)

        try:
            queue_config = QueueConfig.from_dict(config_dict)
        except ConfigValidationError as err:
            errors.append(err)

        try:
            analysis_config = AnalysisConfig.from_dict(config_dict)
        except ConfigValidationError as err:
            errors.append(err)

        if errors:
            raise ConfigValidationError.from_collected(errors)

        env_vars = {}
        for key, val in config_dict.get("SETENV", []):
            env_vars[key] = val

        return cls(
            substitution_list=substitution_list,
            ensemble_config=ensemble_config,
            ens_path=config_dict.get(ConfigKeys.ENSPATH, ErtConfig.DEFAULT_ENSPATH),
            env_vars=env_vars,
            random_seed=config_dict.get(ConfigKeys.RANDOM_SEED),
            analysis_config=analysis_config,
            queue_config=queue_config,
            workflow_jobs=workflow_jobs,
            workflows=workflows,
            hooked_workflows=hooked_workflows,
            runpath_file=Path(runpath_file),
            ert_templates=cls._read_templates(config_dict),
            installed_jobs=installed_jobs,
            forward_model_list=cls.read_forward_model(
                installed_jobs, substitution_list, config_dict
            ),
            model_config=model_config,
            user_config_file=config_file_path,
        )

    @classmethod
    def _log_config_file(cls, config_file: str) -> None:
        """
        Logs what configuration was used to start ert. Because the config
        parsing is quite convoluted we are not able to remove all the comments,
        but the easy ones are filtered out.
        """
        if config_file is not None and os.path.isfile(config_file):
            config_context = ""
            with open(config_file, "r", encoding="utf-8") as file_obj:
                for line in file_obj:
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
                f"Content of the configuration file ({config_file}):\n" + config_context
            )

    @classmethod
    def _log_config_dict(cls, content_dict: Dict[str, Any]) -> None:
        tmp_dict = content_dict.copy()
        tmp_dict.pop("FORWARD_MODEL", None)
        tmp_dict.pop("LOAD_WORKFLOW", None)
        tmp_dict.pop("LOAD_WORKFLOW_JOB", None)
        tmp_dict.pop("HOOK_WORKFLOW", None)
        tmp_dict.pop("WORKFLOW_JOB_DIRECTORY", None)

        logger.info("Content of the config_dict: %s", tmp_dict)

    @staticmethod
    def apply_config_content_defaults(content_dict: dict, config_dir: str):
        if ConfigKeys.ENSPATH not in content_dict:
            content_dict[ConfigKeys.ENSPATH] = os.path.join(
                config_dir, ErtConfig.DEFAULT_ENSPATH
            )
        if ConfigKeys.RUNPATH_FILE not in content_dict:
            content_dict[ConfigKeys.RUNPATH_FILE] = os.path.join(
                config_dir, ErtConfig.DEFAULT_RUNPATH_FILE
            )
        elif not os.path.isabs(content_dict[ConfigKeys.RUNPATH_FILE]):
            content_dict[ConfigKeys.RUNPATH_FILE] = os.path.normpath(
                os.path.join(config_dir, content_dict[ConfigKeys.RUNPATH_FILE])
            )

    @classmethod
    def read_site_config(cls) -> ConfigDict:
        return lark_parse(file=site_config_location(), schema=init_site_config_schema())

    @classmethod
    def read_user_config(cls, user_config_file: str) -> ConfigDict:
        site_config = cls.read_site_config()
        return lark_parse(
            file=user_config_file,
            schema=init_user_config_schema(),
            site_config=site_config,
        )

    @staticmethod
    def check_non_utf_chars(file_path: str) -> None:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
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

    @classmethod
    def _read_templates(cls, config_dict) -> List[Tuple[str, str]]:
        templates = []
        if ConfigKeys.DATA_FILE in config_dict and ConfigKeys.ECLBASE in config_dict:
            # This replicates the behavior of the DATA_FILE implementation
            # in C, it adds the .DATA extension and facilitates magic string
            # replacement in the data file
            source_file = config_dict[ConfigKeys.DATA_FILE]
            target_file = (
                config_dict[ConfigKeys.ECLBASE].replace("%d", "<IENS>") + ".DATA"
            )
            cls.check_non_utf_chars(source_file)
            templates.append([source_file, target_file])

        for template in config_dict.get(ConfigKeys.RUN_TEMPLATE, []):
            templates.append(template)
        return templates

    @classmethod
    def _validate_dict(
        cls, config_dict, config_file: str
    ) -> List[Union[ErrorInfo, ConfigValidationError]]:
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

    @classmethod
    def read_forward_model(
        cls,
        installed_jobs: Dict[str, ExtJob],
        substitution_list: SubstitutionList,
        config_dict,
    ) -> List[ExtJob]:
        errors = []
        jobs = []
        for job_description in config_dict.get(ConfigKeys.FORWARD_MODEL, []):
            if len(job_description) > 1:
                unsubstituted_job_name, args = job_description
            else:
                unsubstituted_job_name = job_description[0]
                args = []
            job_name = substitution_list.substitute(unsubstituted_job_name)
            try:
                job = copy.deepcopy(installed_jobs[job_name])
            except KeyError:
                errors.append(
                    ConfigValidationError.with_context(
                        f"Could not find job {job_name!r} in list"
                        f" of installed jobs: {list(installed_jobs.keys())!r}",
                        job_name,
                    )
                )
                continue
            job.private_args = SubstitutionList()
            for key, val in args:
                job.private_args[key] = val
            jobs.append(job)
        for job_description in config_dict.get(ConfigKeys.SIMULATION_JOB, []):
            try:
                job = copy.deepcopy(installed_jobs[job_description[0]])
            except KeyError:
                errors.append(
                    ConfigValidationError.with_context(
                        f"Could not find job {job_description[0]!r} "
                        "in list of installed jobs.",
                        job_description[0],
                    )
                )
                continue
            job.arglist = job_description[1:]
            jobs.append(job)

        if errors:
            raise ConfigValidationError.from_collected(errors)

        return jobs

    def forward_model_job_name_list(self) -> List[str]:
        return [j.name for j in self.forward_model_list]

    def forward_model_data_to_json(
        self,
        run_id: str,
        iens: int = 0,
        itr: int = 0,
    ) -> Dict[str, Any]:
        context = self.substitution_list

        class Substituter:
            def __init__(self, job):
                job_args = ",".join(
                    [f"{key}={value}" for key, value in job.private_args.items()]
                )
                job_description = f"{job.name}({job_args})"
                self.substitution_context_hint = (
                    f"parsing forward model job `FORWARD_MODEL {job_description}` - "
                    "reconstructed, with defines applied during parsing"
                )
                self.copy_private_args = SubstitutionList()
                for key, val in job.private_args.items():
                    self.copy_private_args[key] = context.substitute_real_iter(
                        val, iens, itr
                    )

            @overload
            def substitute(self, string: str) -> str:
                ...

            @overload
            def substitute(self, string: None) -> None:
                ...

            def substitute(self, string):
                if string is None:
                    return string
                string = self.copy_private_args.substitute(
                    string, self.substitution_context_hint, 1, warn_max_iter=False
                )
                return context.substitute_real_iter(string, iens, itr)

            def filter_env_dict(self, d):
                result = {}
                for key, value in d.items():
                    new_key = self.substitute(key)
                    new_value = self.substitute(value)
                    if new_value is None:
                        result[new_key] = None
                    elif not (new_value[0] == "<" and new_value[-1] == ">"):
                        # Remove values containing "<XXX>". These are expected to be
                        # replaced by substitute, but were not.
                        result[new_key] = new_value
                    else:
                        logger.warning(
                            "Environment variable %s skipped due to"
                            " unmatched define %s",
                            new_key,
                            new_value,
                        )
                # Its expected that empty dicts be replaced with "null"
                # in jobs.json
                if not result:
                    return None
                return result

        def handle_default(job: ExtJob, arg: str) -> str:
            return job.default_mapping.get(arg, arg)

        for job in self.forward_model_list:
            for key, val in job.private_args.items():
                if key in context and key != val:
                    logger.info(
                        f"Private arg '{key}':'{val}' chosen over"
                        f" global '{context[key]}' in forward model {job.name}"
                    )
        config_file_path = (
            Path(self.user_config_file) if self.user_config_file is not None else None
        )
        config_path = str(config_file_path.parent) if config_file_path else ""
        config_file = str(config_file_path.name) if config_file_path else ""
        return {
            "global_environment": self.env_vars,
            "config_path": config_path,
            "config_file": config_file,
            "jobList": [
                {
                    "name": substituter.substitute(job.name),
                    "executable": substituter.substitute(job.executable),
                    "target_file": substituter.substitute(job.target_file),
                    "error_file": substituter.substitute(job.error_file),
                    "start_file": substituter.substitute(job.start_file),
                    "stdout": substituter.substitute(job.stdout_file) + f".{idx}"
                    if job.stdout_file
                    else None,
                    "stderr": substituter.substitute(job.stderr_file) + f".{idx}"
                    if job.stderr_file
                    else None,
                    "stdin": substituter.substitute(job.stdin_file),
                    "argList": [
                        handle_default(job, substituter.substitute(arg))
                        for arg in job.arglist
                    ],
                    "environment": substituter.filter_env_dict(job.environment),
                    "exec_env": substituter.filter_env_dict(job.exec_env),
                    "max_running_minutes": job.max_running_minutes,
                    "max_running": job.max_running,
                    "min_arg": job.min_arg,
                    "arg_types": job.arg_types,
                    "max_arg": job.max_arg,
                }
                for idx, job, substituter in [
                    (idx, job, Substituter(job))
                    for idx, job in enumerate(self.forward_model_list)
                ]
            ],
            "run_id": run_id,
            "ert_pid": str(os.getpid()),
        }

    @classmethod
    def _workflows_from_dict(  # pylint: disable=too-many-branches
        cls,
        content_dict,
        substitution_list,
    ):
        workflow_job_info = content_dict.get(ConfigKeys.LOAD_WORKFLOW_JOB, [])
        workflow_job_dir_info = content_dict.get(ConfigKeys.WORKFLOW_JOB_DIRECTORY, [])
        hook_workflow_info = content_dict.get(ConfigKeys.HOOK_WORKFLOW, [])
        workflow_info = content_dict.get(ConfigKeys.LOAD_WORKFLOW, [])

        workflow_jobs = {}
        workflows = {}
        hooked_workflows = defaultdict(list)

        errors = []

        for workflow_job in workflow_job_info:
            try:
                # WorkflowJob.fromFile only throws error if a
                # non-readable file is provided.
                # Non-existing files are caught by the new parser
                new_job = WorkflowJob.from_file(
                    config_file=workflow_job[0],
                    name=None if len(workflow_job) == 1 else workflow_job[1],
                )
                workflow_jobs[new_job.name] = new_job
            except ErtScriptLoadFailure as err:
                warnings.warn(
                    ConfigWarning.with_context(
                        f"Loading workflow job {workflow_job[0]!r}"
                        f" failed with '{err}'. It will not be loaded.",
                        workflow_job[0],
                    ),
                    stacklevel=1,
                )
            except ConfigValidationError as err:
                errors.append(
                    ErrorInfo(
                        message=str(err).replace("\n", ";"),
                        filename=workflow_job[0],
                    ).set_context(workflow_job[0])
                )

        for job_path in workflow_job_dir_info:
            if not os.path.isdir(job_path):
                warnings.warn(
                    ConfigWarning.with_context(
                        f"Unable to open job directory {job_path}", job_path
                    ),
                    stacklevel=1,
                )
                continue

            files = os.listdir(job_path)
            for file_name in files:
                full_path = os.path.join(job_path, file_name)
                try:
                    new_job = WorkflowJob.from_file(config_file=full_path)
                    workflow_jobs[new_job.name] = new_job
                except ErtScriptLoadFailure as err:
                    warnings.warn(
                        ConfigWarning.with_context(
                            f"Loading workflow job {full_path!r}"
                            f" failed with '{err}'. It will not be loaded.",
                            file_name,
                        ),
                        stacklevel=1,
                    )
                except ConfigValidationError as err:
                    errors.append(
                        ErrorInfo(
                            message=str(err),
                            filename=full_path,
                        ).set_context(job_path)
                    )
        if errors:
            raise ConfigValidationError.from_collected(errors)

        for work in workflow_info:
            filename = os.path.basename(work[0]) if len(work) == 1 else work[1]
            try:
                existed = filename in workflows
                workflows[filename] = Workflow.from_file(
                    work[0],
                    substitution_list,
                    workflow_jobs,
                )
                if existed:
                    warnings.warn(
                        ConfigWarning.with_context(
                            f"Workflow {filename!r} was added twice", work[0]
                        ),
                        stacklevel=1,
                    )
            except ConfigValidationError as err:
                warnings.warn(
                    ConfigWarning.with_context(
                        f"Encountered the following error(s) while "
                        f"reading workflow {filename!r}. It will not be loaded: "
                        + err.cli_message(),
                        work[0],
                    ),
                    stacklevel=1,
                )

        errors = []
        for hook_name, mode_name in hook_workflow_info:
            if hook_name not in workflows:
                errors.append(
                    ErrorInfo(
                        message="Cannot setup hook for non-existing"
                        f" job name {hook_name!r}",
                    ).set_context(hook_name)
                )
                continue

            hooked_workflows[getattr(HookRuntime, mode_name)].append(
                workflows[hook_name]
            )

        if errors:
            raise ConfigValidationError.from_collected(errors)
        return workflow_jobs, workflows, hooked_workflows

    @classmethod
    def _installed_jobs_from_dict(cls, config_dict):
        errors = []
        jobs = {}
        for job in config_dict.get(ConfigKeys.INSTALL_JOB, []):
            name = job[0]
            job_config_file = os.path.abspath(job[1])
            try:
                new_job = ExtJob.from_config_file(
                    name=name,
                    config_file=job_config_file,
                )
            except ConfigValidationError as e:
                errors.append(e)
                continue
            if name in jobs:
                warnings.warn(
                    ConfigWarning.with_context(
                        f"Duplicate forward model job with name {name!r}, choosing "
                        f"{job_config_file!r} over {jobs[name].executable!r}",
                        name,
                    ),
                    stacklevel=1,
                )
            jobs[name] = new_job

        for job_path in config_dict.get(ConfigKeys.INSTALL_JOB_DIRECTORY, []):
            if not os.path.isdir(job_path):
                errors.append(
                    ConfigValidationError.with_context(
                        f"Unable to locate job directory {job_path!r}", job_path
                    )
                )
                continue

            files = os.listdir(job_path)

            if not [
                f
                for f in files
                if os.path.isfile(os.path.abspath(os.path.join(job_path, f)))
            ]:
                warnings.warn(
                    ConfigWarning.with_context(
                        f"No files found in job directory {job_path}", job_path
                    ),
                    stacklevel=1,
                )
                continue

            for file_name in files:
                full_path = os.path.abspath(os.path.join(job_path, file_name))
                if not os.path.isfile(full_path):
                    continue
                try:
                    new_job = ExtJob.from_config_file(config_file=full_path)
                except ConfigValidationError as e:
                    errors.append(e)
                    continue
                name = new_job.name
                if name in jobs:
                    warnings.warn(
                        ConfigWarning.with_context(
                            f"Duplicate forward model job with name {name!r}, "
                            f"choosing {full_path!r} over {jobs[name].executable!r}",
                            name,
                        ),
                        stacklevel=1,
                    )
                jobs[name] = new_job

        if errors:
            raise ConfigValidationError.from_collected(errors)
        return jobs

    def preferred_num_cpu(self) -> int:
        return int(self.substitution_list.get(f"<{ConfigKeys.NUM_CPU}>", 1))
