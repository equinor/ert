import copy
import logging
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Mapping

import pkg_resources

from ert._c_wrappers.config import ConfigParser
from ert._c_wrappers.config.config_parser import ConfigValidationError, ConfigWarning
from ert._c_wrappers.enkf.analysis_config import AnalysisConfig
from ert._c_wrappers.enkf.config_keys import ConfigKeys
from ert._c_wrappers.enkf.ensemble_config import EnsembleConfig
from ert._c_wrappers.enkf.enums import ErtImplType, HookRuntime
from ert._c_wrappers.enkf.model_config import ModelConfig
from ert._c_wrappers.enkf.queue_config import QueueConfig
from ert._c_wrappers.job_queue import (
    ErtScriptLoadFailure,
    ExtJob,
    ExtJobInvalidArgsException,
    Workflow,
    WorkflowJob,
)
from ert._c_wrappers.util import SubstitutionList
from ert._clib import job_kw
from ert._clib.config_keywords import init_site_config_parser, init_user_config_parser

from ._config_content_as_dict import config_content_as_dict
from ._deprecation_migration_suggester import DeprecationMigrationSuggester

logger = logging.getLogger(__name__)


def site_config_location():
    if "ERT_SITE_CONFIG" in os.environ:
        return os.environ["ERT_SITE_CONFIG"]
    return pkg_resources.resource_filename("ert.shared", "share/ert/site-config")


@dataclass
class ErtConfig:
    DEFAULT_ENSPATH: ClassVar[str] = "storage"
    DEFAULT_RUNPATH_FILE: ClassVar[str] = ".ert_runpath_list"

    substitution_list: SubstitutionList = field(default_factory=SubstitutionList)
    ensemble_config: EnsembleConfig = field(default_factory=EnsembleConfig)
    ens_path: str = DEFAULT_ENSPATH
    env_vars: Dict[str, str] = field(default_factory=dict)
    random_seed: str = None
    analysis_config: AnalysisConfig = field(default_factory=AnalysisConfig)
    queue_config: QueueConfig = field(default_factory=QueueConfig)
    workflow_jobs: Dict[str, WorkflowJob] = field(default_factory=dict)
    workflows: Dict[str, Workflow] = field(default_factory=dict)
    hooked_workflows: Dict[HookRuntime, Workflow] = field(default_factory=dict)
    runpath_file: Path = Path(DEFAULT_RUNPATH_FILE)
    ert_templates: List[List[str]] = field(default_factory=list)
    installed_jobs: Dict[str, ExtJob] = field(default_factory=dict)
    forward_model_list: List[ExtJob] = field(default_factory=list)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    user_config_file: str = "no_config"
    config_path: str = field(init=False)

    def __post_init__(self):
        self.config_path = (
            os.path.dirname(os.path.abspath(self.user_config_file))
            if self.user_config_file
            else os.getcwd()
        )

    @classmethod
    def from_file(cls, user_config_file) -> "ErtConfig":
        user_config_dict = ErtConfig.read_user_config(user_config_file)
        config_dir = os.path.abspath(os.path.dirname(user_config_file))
        ErtConfig._log_config_file(user_config_file)
        ErtConfig._log_config_dict(user_config_dict)
        ErtConfig.apply_config_content_defaults(user_config_dict, config_dir)
        return ErtConfig.from_dict(user_config_dict)

    @classmethod
    def from_dict(cls, config_dict) -> "ErtConfig":
        substitution_list = SubstitutionList.from_dict(config_dict=config_dict)
        config_dir = substitution_list.get("<CONFIG_PATH>", "")
        config_file = substitution_list.get("<CONFIG_FILE>", "no_config")
        config_file_path = os.path.join(config_dir, config_file)

        ErtConfig._validate_dict(config_dict, config_file)
        ensemble_config = EnsembleConfig.from_dict(config_dict=config_dict)
        ErtConfig._validate_ensemble_config(ensemble_config, config_file)
        model_config = ModelConfig.from_dict(ensemble_config.refcase, config_dict)
        runpath = model_config.runpath_format_string
        jobname = model_config.jobname_format_string
        substitution_list.addItem("<RUNPATH>", runpath)
        substitution_list.addItem("<ECL_BASE>", jobname)
        substitution_list.addItem("<ECLBASE>", jobname)
        workflow_jobs, workflows, hooked_workflows = ErtConfig._workflows_from_dict(
            config_dict, substitution_list
        )
        installed_jobs = cls._installed_jobs_from_dict(config_dict)
        env_vars = {}
        for key, val in config_dict.get("SETENV", []):
            env_vars[key] = val

        return ErtConfig(
            substitution_list=substitution_list,
            ensemble_config=ensemble_config,
            ens_path=config_dict.get(ConfigKeys.ENSPATH, ErtConfig.DEFAULT_ENSPATH),
            env_vars=env_vars,
            random_seed=config_dict.get(ConfigKeys.RANDOM_SEED, None),
            analysis_config=AnalysisConfig.from_dict(config_dict=config_dict),
            queue_config=QueueConfig.from_dict(config_dict),
            workflow_jobs=workflow_jobs,
            workflows=workflows,
            hooked_workflows=hooked_workflows,
            runpath_file=Path(
                config_dict.get(ConfigKeys.RUNPATH_FILE, ErtConfig.DEFAULT_RUNPATH_FILE)
            ),
            ert_templates=cls._read_templates(config_dict),
            installed_jobs=installed_jobs,
            forward_model_list=cls.read_forward_model(
                installed_jobs, substitution_list, config_dict, config_file
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
    def _create_pre_defines(
        config_file_path: str,
    ) -> Dict[str, str]:
        date_string = date.today().isoformat()
        config_file_dir = os.path.abspath(os.path.dirname(config_file_path))
        config_file_name = os.path.basename(config_file_path)
        config_file_basename = os.path.splitext(config_file_name)[0]
        return {
            "<DATE>": date_string,
            "<CWD>": config_file_dir,
            "<CONFIG_PATH>": config_file_dir,
            "<CONFIG_FILE>": config_file_name,
            "<CONFIG_FILE_BASE>": config_file_basename,
        }

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
    def _create_user_config_parser(cls):
        config_parser = ConfigParser()
        init_user_config_parser(config_parser)
        return config_parser

    @classmethod
    def make_suggestion_list(cls, config_file):
        return DeprecationMigrationSuggester(
            ErtConfig._create_user_config_parser(),
            ErtConfig._create_pre_defines(config_file),
        ).suggest_migrations(config_file)

    @classmethod
    def read_site_config(cls):
        site_config_parser = ConfigParser()
        init_site_config_parser(site_config_parser)
        site_config_content = site_config_parser.parse(site_config_location())
        return config_content_as_dict(site_config_content, {})

    @classmethod
    def read_user_config(cls, user_config_file):
        site_config_dict = ErtConfig.read_site_config()
        user_config_parser = ErtConfig._create_user_config_parser()
        user_config_content = user_config_parser.parse(
            user_config_file,
            pre_defined_kw_map=ErtConfig._create_pre_defines(user_config_file),
        )
        return config_content_as_dict(user_config_content, site_config_dict)

    @classmethod
    def _validate_queue_option_max_running(cls, config_path, config_dict):
        for _, option_name, *values in config_dict.get("QUEUE_OPTION", []):
            if option_name == "MAX_RUNNING" and int(*values) < 0:
                raise ConfigValidationError(
                    errors=[
                        f"QUEUE_OPTION MAX_RUNNING is negative: {str(*values)!r}",
                    ],
                )

    @classmethod
    def _read_templates(cls, config_dict):
        templates = []
        if ConfigKeys.DATA_FILE in config_dict and ConfigKeys.ECLBASE in config_dict:
            # This replicates the behavior of the DATA_FILE implementation
            # in C, it adds the .DATA extension and facilitates magic string
            # replacement in the data file
            source_file = config_dict[ConfigKeys.DATA_FILE]
            target_file = (
                config_dict[ConfigKeys.ECLBASE].replace("%d", "<IENS>") + ".DATA"
            )
            templates.append([source_file, target_file])

        for template in config_dict.get(ConfigKeys.RUN_TEMPLATE, []):
            templates.append(template)
        return templates

    @classmethod
    def _validate_dict(cls, config_dict, config_file):
        if ConfigKeys.JOBNAME in config_dict and ConfigKeys.ECLBASE in config_dict:
            warnings.warn(
                "Can not have both JOBNAME and ECLBASE keywords. "
                "ECLBASE ignored, using JOBNAME with value "
                f"`{config_dict[ConfigKeys.JOBNAME]}` instead",
                category=ConfigWarning,
            )

        if ConfigKeys.SUMMARY in config_dict and ConfigKeys.ECLBASE not in config_dict:
            raise ConfigValidationError(
                "When using SUMMARY keyword, the config must also specify ECLBASE",
                config_file=config_file,
            )
        cls._validate_queue_option_max_running(config_file, config_dict)

    @classmethod
    def _validate_ensemble_config(cls, ensemble_config, config_path):
        for key in ensemble_config.getKeylistFromImplType(ErtImplType.GEN_KW):
            if ensemble_config.getNode(key).getUseForwardInit():
                raise ConfigValidationError(
                    config_file=config_path,
                    errors=[
                        "Loading GEN_KW from files created by the forward model "
                        "is not supported."
                    ],
                )
            if (
                ensemble_config.getNode(key).get_init_file_fmt() is not None
                and "%" not in ensemble_config.getNode(key).get_init_file_fmt()
            ):
                raise ConfigValidationError(
                    config_file=config_path,
                    errors=["Loading GEN_KW from files requires %d in file format"],
                )

    @classmethod
    def read_forward_model(
        cls, installed_jobs, substitution_list, config_dict, config_file
    ):
        jobs = []
        for unsubstituted_job_name, args in config_dict.get(
            ConfigKeys.FORWARD_MODEL, []
        ):
            job_name = substitution_list.substitute(unsubstituted_job_name)
            try:
                job = copy.deepcopy(installed_jobs[job_name])
            except KeyError as err:
                raise ConfigValidationError(
                    errors=(
                        f"Could not find job {job_name!r} in list of installed jobs: "
                        f"{list(installed_jobs.keys())!r}"
                    ),
                    config_file=config_file,
                ) from err
            if args:
                job.private_args = SubstitutionList()
                try:
                    job.private_args.add_from_string(args)
                except ValueError as err:
                    raise ConfigValidationError(
                        errors=f"{err}: 'FORWARD_MODEL {job_name}({args})'\n",
                        config_file=config_file,
                    ) from err

                job.define_args = substitution_list
            try:
                job.validate_args(substitution_list)
            except ExtJobInvalidArgsException as err:
                logger.warning(str(err))
            jobs.append(job)
        for job_description in config_dict.get(ConfigKeys.SIMULATION_JOB, []):
            try:
                job = copy.deepcopy(installed_jobs[job_description[0]])
            except KeyError as err:
                raise ConfigValidationError(
                    f"Could not find job {job_description[0]!r} "
                    "in list of installed jobs.",
                    config_file=config_file,
                ) from err
            job.arglist = job_description[1:]
            job.define_args = substitution_list
            jobs.append(job)

        return jobs

    def forward_model_job_name_list(self) -> List[str]:
        return [j.name for j in self.forward_model_list]

    @staticmethod
    def forward_model_data_to_json(
        forward_model_list: List[ExtJob],
        run_id: str,
        iens: int = 0,
        itr: int = 0,
        context: "SubstitutionList" = None,
        env_varlist: Mapping[str, str] = None,
    ) -> Dict[str, Any]:
        def substitute(job, string: str):
            job_args = ",".join([f"{key}={value}" for key, value in job.private_args])
            job_description = f"{job.name}({job_args})"
            substitution_context_hint = (
                f"parsing forward model job `FORWARD_MODEL {job_description}` - "
                "reconstructed, with defines applied during parsing"
            )
            if string is not None:
                copy_private_args = SubstitutionList()
                for key, val in job.private_args:
                    copy_private_args.addItem(
                        key, context.substitute_real_iter(val, iens, itr)
                    )
                string = copy_private_args.substitute(
                    string, substitution_context_hint, 1
                )
                return context.substitute_real_iter(string, iens, itr)
            else:
                return string

        def handle_default(job: ExtJob, arg: str) -> str:
            return job.default_mapping.get(arg, arg)

        def filter_env_dict(job, d):
            result = {}
            for key, value in d.items():
                new_key = substitute(job, key)
                new_value = substitute(job, value)
                if new_value is None:
                    result[new_key] = None
                elif not (new_value[0] == "<" and new_value[-1] == ">"):
                    # Remove values containing "<XXX>". These are expected to be
                    # replaced by substitute, but were not.
                    result[new_key] = new_value
                else:
                    logger.warning(
                        "Environment variable %s skipped due to unmatched define %s",
                        new_key,
                        new_value,
                    )
            # Its expected that empty dicts be replaced with "null"
            # in jobs.json
            if not result:
                return None
            return result

        if context is None:
            context = SubstitutionList()

        if env_varlist is None:
            env_varlist = {}

        for job in forward_model_list:
            for key, val in job.private_args:
                if key in context and key != val:
                    logger.info(
                        f"Private arg '{key}':'{val}' chosen over"
                        f" global '{context[key]}' in forward model {job.name}"
                    )

        return {
            "global_environment": env_varlist,
            "jobList": [
                {
                    "name": substitute(job, job.name),
                    "executable": substitute(job, job.executable),
                    "target_file": substitute(job, job.target_file),
                    "error_file": substitute(job, job.error_file),
                    "start_file": substitute(job, job.start_file),
                    "stdout": substitute(job, job.stdout_file) + f".{idx}"
                    if job.stdout_file
                    else None,
                    "stderr": substitute(job, job.stderr_file) + f".{idx}"
                    if job.stderr_file
                    else None,
                    "stdin": substitute(job, job.stdin_file),
                    "argList": [
                        handle_default(job, substitute(job, arg)) for arg in job.arglist
                    ],
                    "environment": filter_env_dict(job, job.environment),
                    "exec_env": filter_env_dict(job, job.exec_env),
                    "max_running_minutes": job.max_running_minutes,
                    "max_running": job.max_running,
                    "min_arg": job.min_arg,
                    "arg_types": [
                        job_kw.kw_from_type(int(typ)) for typ in job.arg_types
                    ],
                    "max_arg": job.max_arg,
                }
                for idx, job in enumerate(forward_model_list)
            ],
            "run_id": run_id,
            "ert_pid": str(os.getpid()),
        }

    @classmethod
    def _workflows_from_dict(cls, content_dict, substitution_list):
        workflow_job_info = content_dict.get(ConfigKeys.LOAD_WORKFLOW_JOB, [])
        workflow_job_dir_info = content_dict.get(ConfigKeys.WORKFLOW_JOB_DIRECTORY, [])
        hook_workflow_info = content_dict.get(ConfigKeys.HOOK_WORKFLOW_KEY, [])
        workflow_info = content_dict.get(ConfigKeys.LOAD_WORKFLOW, [])

        workflow_jobs = {}
        workflows = {}
        hooked_workflows = defaultdict(list)

        for workflow_job in workflow_job_info:
            try:
                new_job = WorkflowJob.fromFile(
                    config_file=workflow_job[0],
                    name=None if len(workflow_job) == 1 else workflow_job[1],
                )
                workflow_jobs[new_job.name] = new_job
            except ErtScriptLoadFailure as err:
                warnings.warn(
                    f"Loading workflow job {workflow_job[0]!r} failed with '{err}'."
                    f" It will not be loaded.",
                    category=ConfigWarning,
                )

        for job_path in workflow_job_dir_info:
            if not os.path.isdir(job_path):
                warnings.warn(
                    f"Unable to open job directory {job_path}", category=ConfigWarning
                )
                continue

            files = os.listdir(job_path)
            for file_name in files:
                full_path = os.path.join(job_path, file_name)
                try:
                    new_job = WorkflowJob.fromFile(config_file=full_path)
                    workflow_jobs[new_job.name] = new_job
                except ErtScriptLoadFailure as err:
                    warnings.warn(
                        f"Loading workflow job {full_path!r} failed with '{err}'."
                        f" It will not be loaded.",
                        category=ConfigWarning,
                    )

        for work in workflow_info:
            filename = os.path.basename(work[0]) if len(work) == 1 else work[1]
            try:
                existed = filename in workflows
                workflows[filename] = Workflow.from_file(
                    work[0], substitution_list, workflow_jobs
                )
                if existed:
                    warnings.warn(
                        f"Workflow {filename!r} was added twice",
                        category=ConfigWarning,
                    )
            except ConfigValidationError as err:
                warnings.warn(
                    f"Encountered error(s) {err.errors!r} while"
                    f" reading workflow {filename!r}. It will not be loaded.",
                    category=ConfigWarning,
                )

        for hook_name, mode_name in hook_workflow_info:
            if mode_name not in [runtime.name for runtime in HookRuntime.enums()]:
                raise ConfigValidationError(
                    errors=[f"Run mode {mode_name!r} not supported for Hook Workflow"]
                )

            if hook_name not in workflows:
                raise ConfigValidationError(
                    errors=[
                        f"Cannot setup hook for non-existing job name {hook_name!r}"
                    ]
                )

            hooked_workflows[HookRuntime.from_string(mode_name)].append(
                workflows[hook_name]
            )
        return workflow_jobs, workflows, hooked_workflows

    @classmethod
    def _installed_jobs_from_dict(cls, config_dict):
        jobs = {}
        for job in config_dict.get(ConfigKeys.INSTALL_JOB, []):
            name = job[0]
            job_config_file = os.path.abspath(job[1])
            new_job = cls._create_job(
                job_config_file,
                name,
            )
            if new_job is not None:
                if name in jobs:
                    warnings.warn(
                        f"Duplicate forward model job with name {name!r}, "
                        f"choosing {job_config_file!r} over {jobs[name].executable!r}",
                        category=ConfigWarning,
                    )
                jobs[name] = new_job

        for job_path in config_dict.get(ConfigKeys.INSTALL_JOB_DIRECTORY, []):
            if not os.path.isdir(job_path):
                raise ConfigValidationError(
                    f"Unable to locate job directory {job_path!r}"
                )

            files = os.listdir(job_path)

            if not [
                f
                for f in files
                if os.path.isfile(os.path.abspath(os.path.join(job_path, f)))
            ]:
                warnings.warn(
                    f"No files found in job directory {job_path}",
                    category=ConfigWarning,
                )
                continue

            for file_name in files:
                full_path = os.path.abspath(os.path.join(job_path, file_name))
                new_job = cls._create_job(full_path)
                if new_job is not None:
                    name = new_job.name
                    if name in jobs:
                        warnings.warn(
                            f"Duplicate forward model job with name {name!r}, "
                            f"choosing {full_path!r} over {jobs[name].executable!r}",
                            category=ConfigWarning,
                        )
                    jobs[name] = new_job

        return jobs

    @staticmethod
    def _create_job(job_path, job_name=None):
        if os.path.isfile(job_path):
            return ExtJob.from_config_file(
                name=job_name,
                config_file=job_path,
            )
        return None

    def preferred_num_cpu(self) -> int:
        return int(self.substitution_list.get(f"<{ConfigKeys.NUM_CPU}>", 1))

    def jobname_format_string(self) -> str:
        return self.model_config.jobname_format_string
