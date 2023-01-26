import copy
import logging
import os
import warnings
from collections import defaultdict
from datetime import date
from os.path import isfile
from typing import Any, Dict, List, Mapping, Optional

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


class ResConfig:
    DEFAULT_ENSPATH = "storage"
    DEFAULT_RUNPATH_FILE = ".ert_runpath_list"

    def __init__(
        self,
        user_config_file: Optional[str] = None,
        config_dict: Optional[Dict[ConfigKeys, Any]] = None,
    ):
        self._assert_input(user_config_file, config_dict)
        self.user_config_file = user_config_file

        self._templates = []
        if user_config_file:
            self._alloc_from_content(
                user_config_file=user_config_file,
            )
        else:
            self._alloc_from_dict(config_dict=config_dict)

    def _assert_input(self, user_config_file, config_dict):
        user_config_file_given = user_config_file is not None
        config_dict_given = config_dict is not None
        if user_config_file_given and config_dict_given:
            raise ValueError(
                "Attempting to create ResConfig object with multiple config objects"
            )

        if not user_config_file_given and not config_dict:
            raise ValueError(
                "Error trying to create ResConfig without any configuration"
            )

        if user_config_file_given and not isinstance(user_config_file, str):
            raise ValueError("Expected user_config_file to be a string.")

        if user_config_file_given and not isfile(user_config_file):
            raise IOError(f'No such configuration file "{user_config_file}".')

    def _log_config_file(self, config_file: str) -> None:
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

    def _log_config_dict(self, content_dict: Dict[str, Any]) -> None:
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
    def apply_config_content_defaults(content_dict: dict, config_path: str):
        if ConfigKeys.DATAROOT not in content_dict:
            content_dict[ConfigKeys.DATAROOT] = config_path
        if ConfigKeys.ENSPATH not in content_dict:
            content_dict[ConfigKeys.ENSPATH] = os.path.join(
                config_path, ResConfig.DEFAULT_ENSPATH
            )
        if ConfigKeys.RUNPATH_FILE not in content_dict:
            content_dict[ConfigKeys.RUNPATH_FILE] = os.path.join(
                config_path, ResConfig.DEFAULT_RUNPATH_FILE
            )
        elif not os.path.isabs(content_dict[ConfigKeys.RUNPATH_FILE]):
            content_dict[ConfigKeys.RUNPATH_FILE] = os.path.normpath(
                os.path.join(config_path, content_dict[ConfigKeys.RUNPATH_FILE])
            )

    @classmethod
    def _create_user_config_parser(cls):
        config_parser = ConfigParser()
        init_user_config_parser(config_parser)
        return config_parser

    @classmethod
    def make_suggestion_list(cls, config_file):
        return DeprecationMigrationSuggester(
            ResConfig._create_user_config_parser(),
            ResConfig._create_pre_defines(config_file),
        ).suggest_migrations(config_file)

    @classmethod
    def read_site_config(cls):
        site_config_parser = ConfigParser()
        init_site_config_parser(site_config_parser)
        site_config_content = site_config_parser.parse(site_config_location())
        return config_content_as_dict(site_config_content, {})

    def _validate_queue_option_max_running(self, config_path, config_dict):
        for _, option_name, *values in config_dict.get("QUEUE_OPTION", []):
            if option_name == "MAX_RUNNING" and int(*values) < 0:
                raise ConfigValidationError(
                    config_file=config_path,
                    errors=[
                        f"QUEUE_OPTION MAX_RUNNING is negative: {str(*values)}",
                    ],
                )

    def _alloc_from_content(self, user_config_file):
        site_config_parser = ConfigParser()
        init_site_config_parser(site_config_parser)
        site_config_content = site_config_parser.parse(site_config_location())

        config_parser = ResConfig._create_user_config_parser()
        self.config_path = os.path.abspath(os.path.dirname(user_config_file))
        user_config_content = config_parser.parse(
            user_config_file,
            pre_defined_kw_map=ResConfig._create_pre_defines(user_config_file),
        )

        self._log_config_file(user_config_file)

        config_content_dict = config_content_as_dict(
            user_config_content, site_config_content
        )
        self._log_config_dict(config_content_dict)
        ResConfig.apply_config_content_defaults(config_content_dict, self.config_path)

        self._alloc_from_dict(config_content_dict)

    def _alloc_from_dict(self, config_dict):
        self.ens_path: str = config_dict[ConfigKeys.ENSPATH]
        self.substitution_list = SubstitutionList.from_dict(config_dict=config_dict)
        self.env_vars = {}
        for key, val in config_dict.get("SETENV", []):
            self.env_vars[key] = val
        self.random_seed = config_dict.get(ConfigKeys.RANDOM_SEED, None)
        self.analysis_config = AnalysisConfig.from_dict(config_dict=config_dict)
        self._validate_queue_option_max_running(None, config_dict)
        self.queue_config = QueueConfig.from_dict(config_dict)
        self._workflows_from_dict(config_dict)
        self.ensemble_config = EnsembleConfig.from_dict(config_dict=config_dict)
        self.model_config = ModelConfig.from_dict(
            self.ensemble_config.refcase, config_dict
        )
        runpath = self.model_config.runpath_format_string
        jobname = self.model_config.jobname_format_string
        self.substitution_list.addItem("<RUNPATH>", runpath)
        self.substitution_list.addItem("<ECL_BASE>", jobname)
        self.substitution_list.addItem("<ECLBASE>", jobname)

        if ConfigKeys.DATA_FILE in config_dict and ConfigKeys.ECLBASE in config_dict:
            # This replicates the behavior of the DATA_FILE implementation
            # in C, it adds the .DATA extension and facilitates magic string
            # replacement in the data file
            source_file = config_dict[ConfigKeys.DATA_FILE]
            target_file = (
                config_dict[ConfigKeys.ECLBASE].replace("%d", "<IENS>") + ".DATA"
            )
            self._templates.append([source_file, target_file])

        for template in config_dict.get(ConfigKeys.RUN_TEMPLATE, []):
            self._templates.append(template)

        for key in self.ensemble_config.getKeylistFromImplType(ErtImplType.GEN_KW):
            if self.ensemble_config.getNode(key).getUseForwardInit():
                raise ConfigValidationError(
                    errors=[
                        "Loading GEN_KW from files created by the forward model "
                        "is not supported."
                    ]
                )
            if (
                self.ensemble_config.getNode(key).get_init_file_fmt() is not None
                and "%" not in self.ensemble_config.getNode(key).get_init_file_fmt()
            ):
                raise ConfigValidationError(
                    config_file=self.config_path,
                    errors=["Loading GEN_KW from files requires %d in file format"],
                )

        self.installed_jobs = self._installed_jobs_from_dict(config_dict)
        config_file = (
            self.substitution_list.get("<CONFIG_PATH>", "")
            + "/"
            + self.substitution_list.get("<CONFIG_FILE>", "")
        )
        jobs = []
        for unsubstituted_job_name, args in config_dict.get(
            ConfigKeys.FORWARD_MODEL, []
        ):
            job_name = self.substitution_list.substitute(unsubstituted_job_name)
            try:
                job = copy.deepcopy(self.installed_jobs[job_name])
            except KeyError as err:
                raise ConfigValidationError(
                    errors=(
                        f"Could not find job `{job_name}` in list of installed jobs: "
                        f"{list(self.installed_jobs.keys())}"
                    ),
                    config_file=config_file,
                ) from err
            if args:
                job.private_args = SubstitutionList()
                try:
                    job.private_args.add_from_string(args)
                except ValueError as err:
                    raise ConfigValidationError(
                        errors=f"{err}: FORWARD_MODEL {job_name} ({args})\n",
                        config_file=config_file,
                    ) from err

                job.define_args = self.substitution_list
            try:
                job.validate_args(self.substitution_list)
            except ExtJobInvalidArgsException as err:
                logger.warning(str(err))
            jobs.append(job)

        for job_description in config_dict.get(ConfigKeys.SIMULATION_JOB, []):
            try:
                job = copy.deepcopy(self.installed_jobs[job_description[0]])
            except KeyError as err:
                raise ConfigValidationError(
                    f"Could not find job `{job_description[0]}` "
                    "in list of installed jobs.",
                    config_file=config_file,
                ) from err
            job.arglist = job_description[1:]
            job.define_args = self.substitution_list
            jobs.append(job)

        self.forward_model_list: List[ExtJob] = jobs

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

    def forward_model_job_name_list(self) -> List[str]:
        return [j.name for j in self.forward_model_list]

    @staticmethod
    def forward_model_data_to_json(
        forward_model_list: List[ExtJob],
        run_id: str,
        data_root: str = "data_root",
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
            "DATA_ROOT": data_root,
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
                    "arg_types": [job_kw.kw_from_type(typ) for typ in job.arg_types],
                    "max_arg": job.max_arg,
                }
                for idx, job in enumerate(forward_model_list)
            ],
            "run_id": run_id,
            "ert_pid": str(os.getpid()),
        }

    def _workflows_from_dict(self, content_dict):
        workflow_job_info = content_dict.get(ConfigKeys.LOAD_WORKFLOW_JOB, [])
        workflow_job_dir_info = content_dict.get(ConfigKeys.WORKFLOW_JOB_DIRECTORY, [])
        hook_workflow_info = content_dict.get(ConfigKeys.HOOK_WORKFLOW_KEY, [])
        workflow_info = content_dict.get(ConfigKeys.LOAD_WORKFLOW, [])

        self.workflow_jobs = {}
        self.workflows = {}
        self.hooked_workflows = defaultdict(list)

        for workflow_job in workflow_job_info:
            new_job = WorkflowJob.fromFile(
                config_file=workflow_job[0],
                name=None if len(workflow_job) == 1 else workflow_job[1],
            )
            if new_job is not None:
                self.workflow_jobs[new_job.name] = new_job

        for job_path in workflow_job_dir_info:
            if not os.path.isdir(job_path):
                warnings.warn(
                    f"Unable to open job directory {job_path}", category=ConfigWarning
                )
                continue

            files = os.listdir(job_path)
            for file_name in files:
                full_path = os.path.join(job_path, file_name)
                new_job = WorkflowJob.fromFile(config_file=full_path)
                if new_job is not None:
                    self.workflow_jobs[new_job.name] = new_job

        for work in workflow_info:
            filename = os.path.basename(work[0]) if len(work) == 1 else work[1]
            self.workflows[filename] = Workflow.from_file(
                work[0], self.substitution_list, self.workflow_jobs
            )

        for hook_name, mode_name in hook_workflow_info:
            if mode_name not in [runtime.name for runtime in HookRuntime.enums()]:
                raise ConfigValidationError(
                    errors=[f"Run mode {mode_name} not supported for Hook Workflow"]
                )

            if hook_name not in self.workflows:
                raise ConfigValidationError(
                    errors=[f"Cannot setup hook for non-existing job name {hook_name}"]
                )

            self.hooked_workflows[HookRuntime.from_string(mode_name)].append(
                self.workflows[hook_name]
            )

    @staticmethod
    def _installed_jobs_from_dict(config_dict):
        jobs = {}
        for job in config_dict.get(ConfigKeys.INSTALL_JOB, []):
            name = job[0]
            new_job = ResConfig._create_job(
                os.path.abspath(job[1]),
                name,
            )
            if new_job is not None:
                jobs[name] = new_job

        for job_path in config_dict.get(ConfigKeys.INSTALL_JOB_DIRECTORY, []):
            if not os.path.isdir(job_path):
                raise ConfigValidationError(
                    f"Unable to locate job directory {job_path}"
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
                new_job = ResConfig._create_job(full_path)
                if new_job is not None:
                    name = new_job.name
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

    @property
    def ert_templates(self):
        return self._templates

    def __eq__(self, other):
        return (
            other is not None
            and isinstance(other, ResConfig)
            and self.ens_path == other.ens_path
            and self.substitution_list == other.substitution_list
            and self.installed_jobs == other.installed_jobs
            and self.env_vars == other.env_vars
            and self.random_seed == other.random_seed
            and self.analysis_config == other.analysis_config
            and self.workflow_jobs == other.workflow_jobs
            and self.workflows == other.workflows
            and self.hooked_workflows == other.hooked_workflows
            and self.ert_templates == other.ert_templates
            and self.ensemble_config == other.ensemble_config
            and self.model_config == other.model_config
            and self.queue_config == other.queue_config
        )

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return f"<ResConfig(\n{str(self)}\n)>"

    def __str__(self):
        return (
            f"SubstitutionList: {self.substitution_list},\n"
            f"EnsPath: {self.ens_path},\n"
            f"Installed jobs: {self.installed_jobs},\n"
            f"EnvironmentVarlist: {self.env_vars},\n"
            f"RandomSeed: {self.random_seed},\n"
            f"Num CPUs: {self.preferred_num_cpu()},\n"
            f"AnalysisConfig: {self.analysis_config},\n"
            f"workflow_jobs: {self.workflow_jobs},\n"
            f"hooked_workflows: {self.hooked_workflows},\n"
            f"workflows: {self.workflows},\n"
            f"ErtTemplates: {self.ert_templates},\n"
            f"EnsembleConfig: {self.ensemble_config},\n"
            f"ModelConfig: {self.model_config},\n"
            f"QueueConfig: {self.queue_config}"
        )
