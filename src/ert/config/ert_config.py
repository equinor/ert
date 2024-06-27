# mypy: ignore-errors
import copy
import importlib
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from fnmatch import fnmatch
from os import path
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    no_type_check,
    overload,
)

import xarray as xr
from typing_extensions import Self

from ert.config.gen_data_config import GenDataConfig
from ert.substitution_list import SubstitutionList

from ._get_num_cpu import get_num_cpu_from_data_file
from .analysis_config import AnalysisConfig
from .ensemble_config import EnsembleConfig
from .forward_model_step import (
    ForwardModelStep,
    ForwardModelStepJSON,
    ForwardModelStepPlugin,
    ForwardModelStepValidationError,
)
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
    HookRuntime,
    init_forward_model_schema,
    init_site_config_schema,
    init_user_config_schema,
    lark_parse,
)
from .parsing.observations_parser import (
    GenObsValues,
    HistoryValues,
    ObservationConfigError,
    SummaryValues,
    parse,
)
from .queue_config import QueueConfig
from .summary_config import SummaryConfig
from .workflow import Workflow
from .workflow_job import ErtScriptLoadFailure, WorkflowJob

logger = logging.getLogger(__name__)


def site_config_location() -> str:
    if "ERT_SITE_CONFIG" in os.environ:
        return os.environ["ERT_SITE_CONFIG"]
    return str(
        Path(importlib.util.find_spec("ert.shared").origin).parent
        / "share/ert/site-config"
    )


@dataclass
class ErtConfig:
    DEFAULT_ENSPATH: ClassVar[str] = "storage"
    DEFAULT_RUNPATH_FILE: ClassVar[str] = ".ert_runpath_list"
    PREINSTALLED_FORWARD_MODEL_STEPS: ClassVar[Dict[str, ForwardModelStep]] = {}

    substitution_list: SubstitutionList = field(default_factory=SubstitutionList)
    ensemble_config: EnsembleConfig = field(default_factory=EnsembleConfig)
    ens_path: str = DEFAULT_ENSPATH
    env_vars: Dict[str, str] = field(default_factory=dict)
    random_seed: Optional[int] = None
    analysis_config: AnalysisConfig = field(default_factory=AnalysisConfig)
    queue_config: QueueConfig = field(default_factory=QueueConfig)
    workflow_jobs: Dict[str, WorkflowJob] = field(default_factory=dict)
    workflows: Dict[str, Workflow] = field(default_factory=dict)
    hooked_workflows: Dict[HookRuntime, List[Workflow]] = field(default_factory=dict)
    runpath_file: Path = Path(DEFAULT_RUNPATH_FILE)
    ert_templates: List[Tuple[str, str]] = field(default_factory=list)
    installed_forward_model_steps: Dict[str, ForwardModelStep] = field(
        default_factory=dict
    )
    forward_model_steps: List[ForwardModelStep] = field(default_factory=list)
    summary_keys: List[str] = field(default_factory=list)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    user_config_file: str = "no_config"
    config_path: str = field(init=False)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ErtConfig):
            return False

        return all(getattr(self, attr) == getattr(other, attr) for attr in vars(self))

    def __post_init__(self) -> None:
        self.config_path = (
            path.dirname(path.abspath(self.user_config_file))
            if self.user_config_file
            else os.getcwd()
        )
        self.enkf_obs: EnkfObs = self._create_observations()

        if len(self.summary_keys) != 0:
            self.ensemble_config.addNode(self._create_summary_config())
        self.observations: Dict[str, xr.Dataset] = self.enkf_obs.datasets

    @staticmethod
    def with_plugins(
        forward_model_step_classes: List[Type[ForwardModelStepPlugin]],
    ) -> Type["ErtConfig"]:
        preinstalled_fm_steps: Dict[str, ForwardModelStepPlugin] = {}
        for fm_step_subclass in forward_model_step_classes:
            fm_step = fm_step_subclass()
            preinstalled_fm_steps[fm_step.name] = fm_step

        class ErtConfigWithPlugins(ErtConfig):
            PREINSTALLED_FORWARD_MODEL_STEPS: ClassVar[
                Dict[str, ForwardModelStepPlugin]
            ] = preinstalled_fm_steps

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
        user_config_dict = cls.read_user_config(user_config_file)
        config_dir = path.abspath(path.dirname(user_config_file))
        cls._log_config_file(user_config_file)
        cls._log_config_dict(user_config_dict)
        cls.apply_config_content_defaults(user_config_dict, config_dir)
        return cls.from_dict(user_config_dict)

    @classmethod
    def from_dict(cls, config_dict) -> Self:
        substitution_list = _substitution_list_from_dict(config_dict)
        runpath_file = config_dict.get(
            ConfigKeys.RUNPATH_FILE, ErtConfig.DEFAULT_RUNPATH_FILE
        )
        substitution_list["<RUNPATH_FILE>"] = runpath_file
        config_dir = substitution_list.get("<CONFIG_PATH>", "")
        config_file = substitution_list.get("<CONFIG_FILE>", "no_config")
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
            installed_forward_model_steps = copy.deepcopy(
                cls.PREINSTALLED_FORWARD_MODEL_STEPS
            )

            installed_forward_model_steps.update(
                cls._installed_forward_model_steps_from_dict(config_dict)
            )

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

        try:
            ensemble_config = EnsembleConfig.from_dict(config_dict=config_dict)
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
            summary_keys=cls._read_summary_keys(config_dict, ensemble_config),
            installed_forward_model_steps=installed_forward_model_steps,
            forward_model_steps=cls._create_list_of_forward_model_steps_to_run(
                installed_forward_model_steps,
                substitution_list,
                config_dict,
            ),
            model_config=model_config,
            user_config_file=config_file_path,
        )

    @classmethod
    def _read_summary_keys(
        cls, config_dict, ensemble_config: EnsembleConfig
    ) -> List[str]:
        summary_keys = [
            item
            for sublist in config_dict.get(ConfigKeys.SUMMARY, [])
            for item in sublist
        ]
        optional_keys = []
        refcase_keys = (
            [] if ensemble_config.refcase is None else ensemble_config.refcase.keys
        )
        to_add = list(refcase_keys)
        for key in summary_keys:
            if "*" in key and refcase_keys:
                for i, rkey in list(enumerate(to_add))[::-1]:
                    if fnmatch(rkey, key) and rkey != "TIME":
                        optional_keys.append(rkey)
                        del to_add[i]
            else:
                optional_keys.append(key)

        return optional_keys

    @classmethod
    def _log_config_file(cls, config_file: str) -> None:
        """
        Logs what configuration was used to start ert. Because the config
        parsing is quite convoluted we are not able to remove all the comments,
        but the easy ones are filtered out.
        """
        if config_file is not None and path.isfile(config_file):
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

        gen_kws = config_dict.get("GEN_KW", [])
        for gen_kw in gen_kws:
            file_path = Path(gen_kw[1])
            if file_path.stat().st_size == 0:
                raise ConfigValidationError(f"No parameters specified in {file_path}")

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
    def _create_list_of_forward_model_steps_to_run(
        cls,
        installed_steps: Dict[str, ForwardModelStep],
        substitution_list: SubstitutionList,
        config_dict,
    ) -> List[ForwardModelStep]:
        errors = []
        fm_steps = []
        for fm_step_description in config_dict.get(ConfigKeys.FORWARD_MODEL, []):
            if len(fm_step_description) > 1:
                unsubstituted_step_name, args = fm_step_description
            else:
                unsubstituted_step_name = fm_step_description[0]
                args = []
            fm_step_name = substitution_list.substitute(unsubstituted_step_name)
            try:
                fm_step = copy.deepcopy(installed_steps[fm_step_name])

                # Preserve as ContextString
                fm_step.name = fm_step_name
            except KeyError:
                errors.append(
                    ConfigValidationError.with_context(
                        f"Could not find forward model step {fm_step_name!r} in list"
                        f" of installed forward model steps: {list(installed_steps.keys())!r}",
                        fm_step_name,
                    )
                )
                continue
            fm_step.private_args = SubstitutionList()
            for key, val in args:
                fm_step.private_args[key] = val

            should_add_step = True

            if fm_step.required_keywords:
                for req in fm_step.required_keywords:
                    if req not in fm_step.private_args:
                        errors.append(
                            ConfigValidationError.with_context(
                                f"Required keyword {req} not found for forward model step {fm_step_name}",
                                fm_step_name,
                            )
                        )
                        should_add_step = False

            if should_add_step:
                fm_steps.append(fm_step)

        for fm_step_description in config_dict.get(ConfigKeys.SIMULATION_JOB, []):
            try:
                fm_step = copy.deepcopy(installed_steps[fm_step_description[0]])
            except KeyError:
                errors.append(
                    ConfigValidationError.with_context(
                        f"Could not find forward model step {fm_step_description[0]!r} "
                        f"in list of installed forward model steps: {installed_steps}",
                        fm_step_description[0],
                    )
                )
                continue
            fm_step.arglist = fm_step_description[1:]
            fm_steps.append(fm_step)

        for fm_step in fm_steps:
            if fm_step.name in cls.PREINSTALLED_FORWARD_MODEL_STEPS:
                try:
                    substituted_json = cls._create_forward_model_json(
                        run_id=None,
                        context=substitution_list,
                        forward_model_steps=[fm_step],
                        skip_pre_experiment_validation=True,
                    )
                    job_json = substituted_json["jobList"][0]
                    fm_step.validate_pre_experiment(job_json)
                except ForwardModelStepValidationError as err:
                    errors.append(
                        ConfigValidationError.with_context(
                            f"Forward model step pre-experiment validation failed: {str(err)}",
                            context=fm_step.name,
                        ),
                    )

        if errors:
            raise ConfigValidationError.from_collected(errors)

        return fm_steps

    def forward_model_step_name_list(self) -> List[str]:
        return [j.name for j in self.forward_model_steps]

    def manifest_to_json(self, iens: int = 0, iter: int = 0) -> Dict[str, Any]:
        manifest = {}
        # Add expected parameter files to manifest
        if iter == 0:
            for (
                name,
                parameter_config,
            ) in self.ensemble_config.parameter_configs.items():
                if parameter_config.forward_init and parameter_config.forward_init_file:
                    file_path = parameter_config.forward_init_file.replace(
                        "%d", str(iens)
                    )
                    manifest[name] = file_path
        # Add expected response files to manifest
        for name, respons_config in self.ensemble_config.response_configs.items():
            input_file = str(respons_config.input_file)
            if isinstance(respons_config, SummaryConfig):
                input_file = input_file.replace("<IENS>", str(iens))
                manifest[f"{name}_UNSMRY"] = f"{input_file}.UNSMRY"
                manifest[f"{name}_SMSPEC"] = f"{input_file}.SMSPEC"
            if isinstance(respons_config, GenDataConfig):
                if respons_config.report_steps and iens in respons_config.report_steps:
                    manifest[name] = input_file.replace("%d", str(iens))
                elif "%d" not in input_file:
                    manifest[name] = input_file
        return manifest

    def forward_model_data_to_json(
        self,
        run_id: Optional[str] = None,
        iens: int = 0,
        itr: int = 0,
    ):
        return self._create_forward_model_json(
            context=self.substitution_list,
            forward_model_steps=self.forward_model_steps,
            user_config_file=self.user_config_file,
            env_vars=self.env_vars,
            run_id=run_id,
            iens=iens,
            itr=itr,
        )

    @classmethod
    def _create_forward_model_json(
        cls,
        context: SubstitutionList,
        forward_model_steps: List[ForwardModelStep],
        run_id: Optional[str],
        iens: int = 0,
        itr: int = 0,
        user_config_file: Optional[str] = "",
        env_vars: Optional[Dict[str, str]] = None,
        skip_pre_experiment_validation: bool = False,
    ) -> Dict[str, Any]:
        if env_vars is None:
            env_vars = {}

        class Substituter:
            def __init__(self, fm_step):
                fm_step_args = ",".join(
                    [f"{key}={value}" for key, value in fm_step.private_args.items()]
                )
                fm_step_description = f"{fm_step.name}({fm_step_args})"
                self.substitution_context_hint = (
                    f"parsing forward model step `FORWARD_MODEL {fm_step_description}` - "
                    "reconstructed, with defines applied during parsing"
                )
                self.copy_private_args = SubstitutionList()
                for key, val in fm_step.private_args.items():
                    self.copy_private_args[key] = context.substitute_real_iter(
                        val, iens, itr
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

        def handle_default(fm_step: ForwardModelStep, arg: str) -> str:
            return fm_step.default_mapping.get(arg, arg)

        for fm_step in forward_model_steps:
            for key, val in fm_step.private_args.items():
                if key in context and key != val and context[key] != val:
                    logger.info(
                        f"Private arg '{key}':'{val}' chosen over"
                        f" global '{context[key]}' in forward model step {fm_step.name}"
                    )
        config_file_path = (
            Path(user_config_file) if user_config_file is not None else None
        )
        config_path = str(config_file_path.parent) if config_file_path else ""
        config_file = str(config_file_path.name) if config_file_path else ""

        job_list_errors = []
        job_list: List[ForwardModelStepJSON] = []
        for idx, fm_step in enumerate(forward_model_steps):
            substituter = Substituter(fm_step)
            fm_step_json = {
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
                "environment": substituter.filter_env_dict(fm_step.environment),
                "exec_env": substituter.filter_env_dict(fm_step.exec_env),
                "max_running_minutes": fm_step.max_running_minutes,
            }

            try:
                if not skip_pre_experiment_validation:
                    fm_step_json = fm_step.validate_pre_realization_run(fm_step_json)
            except ForwardModelStepValidationError as exc:
                job_list_errors.append(
                    ErrorInfo(
                        message=f"Validation failed for "
                        f"forward model step {fm_step.name}: {str(exc)}"
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

    @classmethod
    def _workflows_from_dict(
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
                ConfigWarning.ert_context_warn(
                    f"Loading workflow job {workflow_job[0]!r}"
                    f" failed with '{err}'. It will not be loaded.",
                    workflow_job[0],
                )
            except ConfigValidationError as err:
                errors.append(ErrorInfo(message=str(err)).set_context(workflow_job[0]))

        for job_path in workflow_job_dir_info:
            for file_name in _get_files_in_directory(job_path, errors):
                try:
                    new_job = WorkflowJob.from_file(config_file=file_name)
                    workflow_jobs[new_job.name] = new_job
                except ErtScriptLoadFailure as err:
                    ConfigWarning.ert_context_warn(
                        f"Loading workflow job {file_name!r}"
                        f" failed with '{err}'. It will not be loaded.",
                        file_name,
                    )
                except ConfigValidationError as err:
                    errors.append(ErrorInfo(message=str(err)).set_context(job_path))
        if errors:
            raise ConfigValidationError.from_collected(errors)

        for work in workflow_info:
            filename = path.basename(work[0]) if len(work) == 1 else work[1]
            try:
                existed = filename in workflows
                workflow = Workflow.from_file(
                    work[0],
                    substitution_list,
                    workflow_jobs,
                )
                for job, args in workflow:
                    if job.ert_script:
                        try:
                            job.ert_script.validate(args)
                        except ConfigValidationError as err:
                            errors.append(
                                ErrorInfo(message=(str(err))).set_context(work[0])
                            )
                            continue
                workflows[filename] = workflow
                if existed:
                    ConfigWarning.ert_context_warn(
                        f"Workflow {filename!r} was added twice", work[0]
                    )
            except ConfigValidationError as err:
                ConfigWarning.ert_context_warn(
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

            hooked_workflows[mode].append(workflows[hook_name])

        if errors:
            raise ConfigValidationError.from_collected(errors)
        return workflow_jobs, workflows, hooked_workflows

    @classmethod
    def _installed_forward_model_steps_from_dict(
        cls, config_dict
    ) -> Dict[str, ForwardModelStep]:
        errors = []
        fm_steps = {}
        for fm_step in config_dict.get(ConfigKeys.INSTALL_JOB, []):
            name = fm_step[0]
            fm_step_config_file = path.abspath(fm_step[1])
            try:
                new_fm_step = _forward_model_step_from_config_file(
                    name=name,
                    config_file=fm_step_config_file,
                )
            except ConfigValidationError as e:
                errors.append(e)
                continue
            if name in fm_steps:
                ConfigWarning.ert_context_warn(
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
                    new_fm_step = _forward_model_step_from_config_file(
                        config_file=file_name
                    )
                except ConfigValidationError as e:
                    errors.append(e)
                    continue
                name = new_fm_step.name
                if name in fm_steps:
                    ConfigWarning.ert_context_warn(
                        f"Duplicate forward model step with name {name!r}, "
                        f"choosing {file_name!r} over {fm_steps[name].executable!r}",
                        name,
                    )
                fm_steps[name] = new_fm_step

        if errors:
            raise ConfigValidationError.from_collected(errors)
        return fm_steps

    @property
    def preferred_num_cpu(self) -> int:
        return int(self.substitution_list.get(f"<{ConfigKeys.NUM_CPU}>", 1))

    def _create_summary_config(self) -> SummaryConfig:
        if self.ensemble_config.eclbase is None:
            raise ConfigValidationError(
                "In order to use summary responses, ECLBASE has to be set."
            )
        time_map = (
            set(self.ensemble_config.refcase.dates)
            if self.ensemble_config.refcase is not None
            else None
        )
        return SummaryConfig(
            name="summary",
            input_file=self.ensemble_config.eclbase,
            keys=self.summary_keys,
            refcase=time_map,
        )

    def _create_observations(self) -> EnkfObs:
        obs_vectors: Dict[str, ObsVector] = {}
        obs_config_file = self.model_config.obs_config_file
        obs_time_list: Sequence[datetime] = []
        if self.ensemble_config.refcase is not None:
            obs_time_list = self.ensemble_config.refcase.all_dates
        elif self.model_config.time_map is not None:
            obs_time_list = self.model_config.time_map
        if obs_config_file:
            if path.isfile(obs_config_file) and path.getsize(obs_config_file) == 0:
                raise ObservationConfigError.with_context(
                    f"Empty observations file: {obs_config_file}", obs_config_file
                )

            if not os.access(obs_config_file, os.R_OK):
                raise ObservationConfigError.with_context(
                    "Do not have permission to open observation"
                    f" config file {obs_config_file!r}",
                    obs_config_file,
                )
            obs_config_content = parse(obs_config_file)
            history = self.model_config.history_source
            std_cutoff = self.analysis_config.std_cutoff
            time_len = len(obs_time_list)
            ensemble_config = self.ensemble_config
            config_errors: List[ErrorInfo] = []
            for obs_name, values in obs_config_content:
                try:
                    if type(values) == HistoryValues:
                        self.summary_keys.append(obs_name)
                        obs_vectors.update(
                            **EnkfObs._handle_history_observation(
                                ensemble_config,
                                values,
                                obs_name,
                                std_cutoff,
                                history,
                                time_len,
                            )
                        )
                    elif type(values) == SummaryValues:
                        self.summary_keys.append(values.key)
                        obs_vectors.update(
                            **EnkfObs._handle_summary_observation(
                                values,
                                obs_name,
                                obs_time_list,
                                bool(ensemble_config.refcase),
                            )
                        )
                    elif type(values) == GenObsValues:
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
                                message=f"Unknown ObservationType {type(values)} for {obs_name}"
                            ).set_context(obs_name)
                        )
                        continue
                except ObservationConfigError as err:
                    config_errors.extend(err.errors)
                except ValueError as err:
                    config_errors.append(
                        ErrorInfo(message=str(err)).set_context(obs_name)
                    )

            if config_errors:
                raise ObservationConfigError.from_collected(config_errors)

        return EnkfObs(obs_vectors, obs_time_list)


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
        ConfigWarning.ert_context_warn(
            f"No files found in job directory {job_path}", job_path
        )
    return files


def _substitution_list_from_dict(config_dict) -> SubstitutionList:
    subst_list = SubstitutionList()

    for key, val in config_dict.get("DEFINE", []):
        subst_list[key] = val

    if "<CONFIG_PATH>" not in subst_list:
        subst_list["<CONFIG_PATH>"] = config_dict.get("CONFIG_DIRECTORY", os.getcwd())

    num_cpus = config_dict.get("NUM_CPU")
    if num_cpus is None and "DATA_FILE" in config_dict:
        num_cpus = get_num_cpu_from_data_file(config_dict.get("DATA_FILE"))
    if num_cpus is None:
        num_cpus = 1
    subst_list["<NUM_CPU>"] = str(num_cpus)

    for key, val in config_dict.get("DATA_KW", []):
        subst_list[key] = val

    return subst_list


@no_type_check
def _forward_model_step_from_config_file(
    config_file: str, name: Optional[str] = None
) -> "ForwardModelStep":
    if name is None:
        name = os.path.basename(config_file)

    schema = init_forward_model_schema()

    try:
        content_dict = lark_parse(file=config_file, schema=schema, pre_defines=[])

        specified_arg_types: List[Tuple[int, str]] = content_dict.get(
            ForwardModelStepKeys.ARG_TYPE, []
        )

        specified_max_args: int = content_dict.get("MAX_ARG", 0)
        specified_min_args: int = content_dict.get("MIN_ARG", 0)

        arg_types_list = parse_arg_types_list(
            specified_arg_types, specified_min_args, specified_max_args
        )

        environment = {k: v for [k, v] in content_dict.get("ENV", [])}
        exec_env = {k: v for [k, v] in content_dict.get("EXEC_ENV", [])}
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
            exec_env=exec_env,
            default_mapping=default_mapping,
            help_text=content_dict.get("HELP_TEXT", ""),
        )
    except IOError as err:
        raise ConfigValidationError.with_context(str(err), config_file) from err
