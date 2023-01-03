import copy
import logging
import os
import sys
from datetime import date
from os.path import isfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ecl.util.util import StringList

from ert._c_wrappers.config import ConfigContent, ConfigParser
from ert._c_wrappers.config.config_parser import ConfigValidationError
from ert._c_wrappers.enkf.analysis_config import AnalysisConfig
from ert._c_wrappers.enkf.config_keys import ConfigKeys
from ert._c_wrappers.enkf.ensemble_config import EnsembleConfig
from ert._c_wrappers.enkf.enums import ErtImplType
from ert._c_wrappers.enkf.ert_workflow_list import ErtWorkflowList
from ert._c_wrappers.enkf.model_config import ModelConfig
from ert._c_wrappers.enkf.queue_config import QueueConfig
from ert._c_wrappers.job_queue import EnvironmentVarlist, ExtJob, ForwardModel
from ert._c_wrappers.util import SubstitutionList
from ert._clib.config_keywords import init_site_config_parser, init_user_config_parser

from ._config_content_as_dict import config_content_as_dict
from ._deprecation_migration_suggester import DeprecationMigrationSuggester

logger = logging.getLogger(__name__)


def site_config_location():

    if "ERT_SITE_CONFIG" in os.environ:
        return os.environ["ERT_SITE_CONFIG"]

    path = Path(__file__).parent
    for p in path.parents:
        npath = p / "ert" / "shared" / "share" / "ert" / "site-config"
        if npath.is_file():
            path = npath
            break
    else:
        raise SystemError("Could not find `share/ert/site-config`")
    return str(path)


def parse_signature_job(signature: str) -> Tuple[str, Optional[str]]:
    """Parses the job description as a signature type job.

    A signature is on the form job(arg1=val1, arg2=val2, arg3=val3). This is used
    in the FORWARD_MODEL keyword:

        FORWARD_MODEL job(arg1=val1, arg2=val2, arg3=val3)

    :returns: Tuple of the job name, and the string of argument assignments.


    >>> parse_signature_job("job(arg1=val1, arg2=val2, arg3=val3)")
    ('job', 'arg1=val1, arg2=val2, arg3=val3')

    Function without arguments has arglist set to None:

    >>> parse_signature_job("job")
    ('job', None)


    For backwards compatability, text after first closing parenthesis is closed,
    but a warning is displayed.
    """

    open_paren = signature.find("(")
    if open_paren == -1:
        return signature, None
    job_name = signature[:open_paren]
    close_paren = signature.find(")")
    if close_paren == -1:
        raise ConfigValidationError(
            errors=[f"Missing ) in FORWARD_MODEL job description {signature}"]
        )
    if close_paren < len(signature) - 1:
        logger.warning(
            f'Arguments after closing ) in "{signature}"'
            f' ("{signature[close_paren:]}") is ignored.'
        )
    return job_name, signature[open_paren + 1 : close_paren]


class ResConfig:
    def __init__(
        self,
        user_config_file: Optional[str] = None,
        config: Optional[Dict[ConfigKeys, Any]] = None,
        config_dict: Optional[Dict[ConfigKeys, Any]] = None,
    ):

        self._assert_input(user_config_file, config, config_dict)
        self.user_config_file = user_config_file

        self._errors, self._failed_keys = None, None
        self._templates = []
        if user_config_file or config:
            self._alloc_from_content(
                user_config_file=user_config_file,
                config=config,
            )
        else:
            self._alloc_from_dict(config_dict=config_dict)

        self._validate()

    def _assert_input(self, user_config_file, config, config_dict):
        configs = sum(
            1 for x in [user_config_file, config, config_dict] if x is not None
        )

        if configs > 1:
            raise ValueError(
                "Attempting to create ResConfig object with multiple config objects"
            )

        if configs == 0:
            raise ValueError(
                "Error trying to create ResConfig without any configuration"
            )

        if config and not isinstance(config, dict):
            raise ValueError(f"Expected config to be a dictionary, was {type(config)}")

        if user_config_file and not isinstance(user_config_file, str):
            raise ValueError("Expected user_config_file to be a string.")

        if user_config_file is not None and not isfile(user_config_file):
            raise IOError(f'No such configuration file "{user_config_file}".')

    def _validate(self):
        for key, _ in self.substitution_list:
            if (
                key.count("<") != 1
                or key.count(">") != 1
                or key[0] != "<"
                or key[-1] != ">"
            ):
                print(
                    "Using DEFINE or DATA_KW with substitution"
                    " strings that are not of the form '<KEY>' is deprecated"
                    " and can result in undefined behavior. "
                    f"Please change {key} to <{key.replace('<', '').replace('>', '')}>",
                    file=sys.stderr,
                )

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

    def _log_config_content(self, config_content: ConfigContent) -> None:
        tmp_dict = config_content_as_dict(config_content, {}).copy()
        tmp_dict.pop("FORWARD_MODEL", None)
        tmp_dict.pop("LOAD_WORKFLOW", None)
        tmp_dict.pop("LOAD_WORKFLOW_JOB", None)
        tmp_dict.pop("HOOK_WORKFLOW", None)
        tmp_dict.pop("WORKFLOW_JOB_DIRECTORY", None)

        logger.info("Content of the config_content:")
        logger.info(tmp_dict)

    @staticmethod
    def _create_pre_defines(
        config_file_path: str,
    ) -> dict:
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
                config_path, ModelConfig.DEFAULT_ENSPATH
            )

    @classmethod
    def _create_user_config_parser(cls):
        config_parser = ConfigParser()
        init_user_config_parser(config_parser)
        return config_parser

    def _display_suggestions(self, config_file):
        suggestions = DeprecationMigrationSuggester(
            ResConfig._create_user_config_parser()
        ).suggest_migrations(config_file)
        for suggestion in suggestions:
            logging.error(suggestion)

    @classmethod
    def make_suggestion_list(cls, config_file):
        return DeprecationMigrationSuggester(
            ResConfig._create_user_config_parser()
        ).suggest_migrations(config_file)

    # build configs from config file or everest dict
    def _alloc_from_content(self, user_config_file=None, config=None):
        site_config_parser = ConfigParser()
        init_site_config_parser(site_config_parser)
        site_config_content = site_config_parser.parse(site_config_location())
        if user_config_file is not None:
            self._display_suggestions(user_config_file)
            # initialize configcontent if user_file provided
            config_parser = ResConfig._create_user_config_parser()
            init_user_config_parser(config_parser)
            self.config_path = os.path.abspath(os.path.dirname(user_config_file))
            user_config_content = config_parser.parse(
                user_config_file,
                pre_defined_kw_map=ResConfig._create_pre_defines(user_config_file),
            )
        else:
            self.config_path = os.getcwd()
            user_config_content = self._build_config_content(config)

        self._log_config_file(user_config_file)
        self._log_config_content(user_config_content)

        if self.errors:
            logging.error(f"Error loading configuration: {str(self._errors)}")
            raise ConfigValidationError(
                config_file=user_config_file, errors=self._errors
            )

        config_content_dict = config_content_as_dict(
            user_config_content, site_config_content
        )
        ResConfig.apply_config_content_defaults(config_content_dict, self.config_path)
        self.substitution_list = SubstitutionList.from_dict(config_content_dict)

        self.env_vars = EnvironmentVarlist.from_dict(config_content_dict)
        self.random_seed = config_content_dict.get(ConfigKeys.RANDOM_SEED, None)
        self.analysis_config = AnalysisConfig.from_dict(config_content_dict)
        self.queue_config = QueueConfig.from_dict(config_content_dict)

        self.ert_workflow_list = ErtWorkflowList.from_dict(config_content_dict)
        self.runpath_file = self.substitution_list[f"<{ConfigKeys.RUNPATH_FILE}>"]

        if user_config_content.hasKey(
            ConfigKeys.DATA_FILE
        ) and user_config_content.hasKey(ConfigKeys.ECLBASE):
            # This replicates the behavior of the DATA_FILE implementation
            # in C, it adds the .DATA extension and facilitates magic string
            # replacement in the data file
            source_file = user_config_content[ConfigKeys.DATA_FILE]
            target_file = user_config_content[ConfigKeys.ECLBASE]
            target_file = target_file.getValue(0).replace("%d", "<IENS>")
            self._templates.append([source_file.getValue(0), target_file + ".DATA"])

        if user_config_content.hasKey(ConfigKeys.RUN_TEMPLATE):
            for template in user_config_content[ConfigKeys.RUN_TEMPLATE]:
                self._templates.append(list(template))

        self.ensemble_config = EnsembleConfig.from_dict(config_dict=config_content_dict)

        for key in self.ensemble_config.getKeylistFromImplType(ErtImplType.GEN_KW):
            if self.ensemble_config.getNode(key).getUseForwardInit():
                raise KeyError(
                    "Loading GEN_KW from files created by the forward model "
                    "is not supported."
                )
            if (
                self.ensemble_config.getNode(key).get_init_file_fmt() != None
                and "%" not in self.ensemble_config.getNode(key).get_init_file_fmt()
            ):
                raise ConfigValidationError(
                    config_file=self.config_path,
                    errors=["Loading GEN_KW from files requires %d in file format"],
                )

        self.installed_jobs = self._installed_jobs_from_dict(config_content_dict)
        jobs = []
        # FORWARD_MODEL_KEY
        for job_description in config_content_dict.get(ConfigKeys.FORWARD_MODEL, []):
            job_name, args = parse_signature_job("".join(job_description))
            try:
                job = copy.deepcopy(self.installed_jobs[job_name])
            except KeyError as err:
                raise ValueError(
                    f"Could not find job `{job_name}` in list of installed jobs: "
                    f"{list(self.installed_jobs.keys())}"
                ) from err
            if args is not None:
                job.private_args = SubstitutionList()
                job.private_args.add_from_string(args)
                job.define_args = self.substitution_list
            jobs.append(job)

        # SIMULATION_JOB_KEY
        for job_description in config_content_dict.get(ConfigKeys.SIMULATION_JOB, []):
            job = copy.deepcopy(self.installed_jobs[job_description[0]])
            job.arglist = job_description[1:]
            job.define_args = self.substitution_list
            jobs.append(job)

        self.forward_model = ForwardModel(jobs=jobs)

        if (
            ConfigKeys.JOBNAME in config_content_dict
            and ConfigKeys.ECLBASE in config_content_dict
        ):
            logger.warning(
                "Can not have both JOBNAME and ECLBASE keywords. "
                "ECLBASE ignored, using JOBNAME with value "
                f"`{config_content_dict[ConfigKeys.JOBNAME]}` instead"
            )

        if (
            ConfigKeys.SUMMARY in config_content_dict
            and ConfigKeys.ECLBASE not in config_content_dict
        ):
            raise ConfigValidationError(
                "When using SUMMARY keyword, the config must also specify ECLBASE"
            )

        self.model_config = ModelConfig.from_dict(
            self.ensemble_config.refcase, config_content_dict
        )

    # build configs from config dict
    def _alloc_from_dict(self, config_dict):
        site_config_parser = ConfigParser()
        init_site_config_parser(site_config_parser)
        # treat the default config dir

        self.substitution_list = SubstitutionList.from_dict(config_dict=config_dict)
        self.env_vars = EnvironmentVarlist.from_dict(config_dict=config_dict)
        self.random_seed = config_dict.get(ConfigKeys.RANDOM_SEED, None)
        self.analysis_config = AnalysisConfig.from_dict(config_dict=config_dict)
        self.queue_config = QueueConfig.from_dict(config_dict)

        self.ert_workflow_list = ErtWorkflowList.from_dict(config_dict)

        if ConfigKeys.DATA_FILE in config_dict and ConfigKeys.ECLBASE in config_dict:
            # This replicates the behavior of the DATA_FILE implementation
            # in C, it adds the .DATA extension and facilitates magic string
            # replacement in the data file
            source_file = config_dict[ConfigKeys.DATA_FILE]
            target_file = config_dict[ConfigKeys.ECLBASE].replace("%d", "<IENS>")
            self._templates.append(
                [os.path.abspath(source_file), target_file + ".DATA"]
            )

        self.runpath_file = config_dict.get(
            ConfigKeys.RUNPATH_FILE, ".ert_runpath_list"
        )
        templates = config_dict.get(ConfigKeys.RUN_TEMPLATE, [])
        for source_file, target_file, *_ in templates:
            self._templates.append([os.path.abspath(source_file), target_file])

        self.ensemble_config = EnsembleConfig.from_dict(config_dict=config_dict)

        for key in self.ensemble_config.getKeylistFromImplType(ErtImplType.GEN_KW):
            if self.ensemble_config.getNode(key).getUseForwardInit():
                raise ConfigValidationError(
                    errors=[
                        "Loading GEN_KW from files created by the forward model "
                        "is not supported."
                    ]
                )
            if (
                self.ensemble_config.getNode(key).get_init_file_fmt() != None
                and "%" not in self.ensemble_config.getNode(key).get_init_file_fmt()
            ):
                raise ConfigValidationError(
                    errors=["Loading GEN_KW from files requires %d in file format"]
                )

        self.installed_jobs = self._installed_jobs_from_dict(config_dict)
        jobs = []
        # FORWARD_MODEL_KEY
        for job_description in config_dict.get(ConfigKeys.FORWARD_MODEL, []):
            try:
                job = copy.deepcopy(
                    self.installed_jobs[job_description[ConfigKeys.NAME]]
                )
            except KeyError as err:
                raise ValueError(
                    f"Could not find job `{job_description[ConfigKeys.NAME]}`"
                    f" in list of installed jobs: {list(self.installed_jobs.keys())}"
                ) from err
            job.private_args = SubstitutionList()
            job.private_args.add_from_string(job_description.get(ConfigKeys.ARGLIST))
            jobs.append(job)

        # SIMULATION_JOB_KEY
        for job_description in config_dict.get(ConfigKeys.SIMULATION_JOB, []):
            job = copy.deepcopy(self.installed_jobs[job_description[ConfigKeys.NAME]])
            try:
                job = copy.deepcopy(
                    self.installed_jobs[job_description[ConfigKeys.NAME]]
                )
            except KeyError as err:
                raise ValueError(
                    f"Could not find job `{job_description[ConfigKeys.NAME]}` "
                    "in list of installed jobs."
                ) from err
            job.private_args = SubstitutionList()
            job.private_args.add_from_string(job_description.get(ConfigKeys.ARGLIST))
            jobs.append(job)

        self.forward_model = ForwardModel(jobs=jobs)

        if ConfigKeys.JOBNAME in config_dict and ConfigKeys.ECLBASE in config_dict:
            logger.warning(
                "Can not have both JOBNAME and ECLBASE keywords. "
                "ECLBASE ignored, using JOBNAME with value "
                f"`{config_dict[ConfigKeys.JOBNAME]}` instead"
            )

        if ConfigKeys.SUMMARY in config_dict and ConfigKeys.ECLBASE not in config_dict:
            raise ConfigValidationError(
                "When using SUMMARY keyword, the config must also specify ECLBASE"
            )

        self.model_config = ModelConfig.from_dict(
            self.ensemble_config.refcase, config_dict
        )

    @staticmethod
    def _installed_jobs_from_dict(config_dict):
        jobs = {}
        # fill in joblist
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
                logger.warning(f"Unable to locate job directory {job_path}")
                continue

            files = os.listdir(job_path)

            if not [
                f
                for f in files
                if os.path.isfile(os.path.abspath(os.path.join(job_path, f)))
            ]:
                logger.warning(f"No files found in job directory {job_path}")
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

    def _extract_defines(self, config):
        defines = {}
        if ConfigKeys.DEFINES in config:
            for key in config[ConfigKeys.DEFINES]:
                defines[key] = str(config[ConfigKeys.DEFINES][key])

        return defines

    def _parse_value(self, value):
        if isinstance(value, str):
            return value
        elif isinstance(value, list):
            return [str(elem) for elem in value]
        else:
            return str(value)

    def _assert_keys(self, mother_key, exp_keys, keys):
        if set(exp_keys) != set(keys):
            raise ValueError(
                f"Did expect the keys {exp_keys} in {mother_key}, received {keys}."
            )

    def _extract_internals(self, config):
        internal_config = []
        config_dir = os.getcwd()

        if ConfigKeys.INTERNALS in config:
            intercon = config[ConfigKeys.INTERNALS]

            dir_key = ConfigKeys.CONFIG_DIRECTORY
            if dir_key in intercon:
                config_dir = os.path.realpath(intercon[dir_key])

            internal_filter = [dir_key]
            for key, value in intercon.items():
                if key not in internal_filter:
                    internal_config.append((key, self._parse_value(value)))

        internal_config.append((ConfigKeys.CONFIG_DIRECTORY, config_dir))
        return config_dir, internal_config

    def _extract_queue_system(self, config):
        if ConfigKeys.QUEUE_SYSTEM not in config:
            return []

        qc = config[ConfigKeys.QUEUE_SYSTEM]
        queue_config = []
        if ConfigKeys.QUEUE_OPTION in qc:
            for qo in qc[ConfigKeys.QUEUE_OPTION]:
                queue_options = [
                    ConfigKeys.DRIVER_NAME,
                    ConfigKeys.OPTION,
                    ConfigKeys.VALUE,
                ]

                self._assert_keys(ConfigKeys.QUEUE_OPTION, queue_options, qo.keys())

                value = [str(qo[item]) for item in queue_options]
                queue_config.append((ConfigKeys.QUEUE_OPTION, value))

        queue_system_filter = [ConfigKeys.QUEUE_OPTION]
        for key, value in qc.items():
            if key not in queue_system_filter:
                queue_config.append((key, self._parse_value(value)))

        return queue_config

    def _extract_install_job(self, config):
        if ConfigKeys.INSTALL_JOB not in config:
            return []

        ic = config[ConfigKeys.INSTALL_JOB]
        job_config = []
        for job in ic:
            job_options = [ConfigKeys.NAME, ConfigKeys.PATH]

            self._assert_keys(ConfigKeys.INSTALL_JOB, job_options, job.keys())
            value = [str(job[item]) for item in job_options]
            job_config.append((ConfigKeys.INSTALL_JOB, value))

        return job_config

    def _extract_simulation_job(self, config):
        if ConfigKeys.SIMULATION_JOB not in config:
            return []

        ic = config[ConfigKeys.SIMULATION_JOB]
        simulation_job = []
        for job in ic:
            arglist = [job[ConfigKeys.NAME]]
            if ConfigKeys.ARGLIST in job:
                for arg in job[ConfigKeys.ARGLIST]:
                    arglist.append(str(arg))
            simulation_job.append((ConfigKeys.SIMULATION_JOB, arglist))

        return simulation_job

    def _extract_forward_model(self, config):
        if ConfigKeys.FORWARD_MODEL not in config:
            return []

        ic = config[ConfigKeys.FORWARD_MODEL]
        forward_model_job = []
        for job in ic:
            forward_model_job.append((ConfigKeys.FORWARD_MODEL, job))

        return forward_model_job

    def _extract_logging(self, config):
        if ConfigKeys.LOGGING not in config:
            return []

        logging_config = []
        for key, value in config[ConfigKeys.LOGGING].items():
            logging_config.append((key, self._parse_value(value)))

        return logging_config

    def _extract_seed(self, config):
        if ConfigKeys.SEED not in config:
            return []

        seed_config = []
        for key, value in config[ConfigKeys.SEED].items():
            seed_config.append((key, self._parse_value(value)))

        return seed_config

    def _extract_gen_kw(self, config):
        if ConfigKeys.GEN_KW not in config:
            return []

        gen_kw_config = []
        for gk in config[ConfigKeys.GEN_KW]:
            gen_kw_options = [
                ConfigKeys.NAME,
                ConfigKeys.TEMPLATE,
                ConfigKeys.OUT_FILE,
                ConfigKeys.PARAMETER_FILE,
            ]

            self._assert_keys(ConfigKeys.GEN_KW, gen_kw_options, gk.keys())

            value = [gk[item] for item in gen_kw_options]
            gen_kw_config.append((ConfigKeys.GEN_KW, value))

        return gen_kw_config

    def _extract_gen_data(self, config):
        if ConfigKeys.GEN_DATA not in config:
            return []

        gen_data_config = []
        for gd in config[ConfigKeys.GEN_DATA]:
            req_keys = [
                ConfigKeys.NAME,
                ConfigKeys.RESULT_FILE,
                ConfigKeys.REPORT_STEPS,
            ]

            default_opt = {ConfigKeys.INPUT_FORMAT: "ASCII"}

            if not sorted(req_keys) == sorted(gd.keys()):
                raise KeyError(
                    f"Expected keys {req_keys} when creating GEN_DATA, "
                    f"received {gd}"
                )

            value = [gd[ConfigKeys.NAME]]
            value += [f"{key}:{gd[key]}" for key in req_keys[1:]]
            value += [f"{key}:{val}" for key, val in default_opt.items()]
            gen_data_config.append((ConfigKeys.GEN_DATA, value))

        return gen_data_config

    def _extract_simulation(self, config):
        if ConfigKeys.SIMULATION not in config:
            return []

        simulation_config = []
        sc = config[ConfigKeys.SIMULATION]
        sim_filter = []

        # Extract queue system
        sim_filter.append(ConfigKeys.QUEUE_SYSTEM)
        simulation_config += self._extract_queue_system(sc)

        # Extract install job
        sim_filter.append(ConfigKeys.INSTALL_JOB)
        simulation_config += self._extract_install_job(sc)

        # Extract forward_model
        sim_filter.append(ConfigKeys.FORWARD_MODEL)
        simulation_config += self._extract_forward_model(sc)

        # Extract simulation_job
        sim_filter.append(ConfigKeys.SIMULATION_JOB)
        simulation_config += self._extract_simulation_job(sc)

        # Extract logging
        sim_filter.append(ConfigKeys.LOGGING)
        simulation_config += self._extract_logging(sc)

        # Extract seed
        sim_filter.append(ConfigKeys.SEED)
        simulation_config += self._extract_seed(sc)

        # Extract GEN_KW
        sim_filter.append(ConfigKeys.GEN_KW)
        simulation_config += self._extract_gen_kw(sc)

        # Extract GEN_DATA
        sim_filter.append(ConfigKeys.GEN_DATA)
        simulation_config += self._extract_gen_data(sc)

        # Others
        for key, value in sc.items():
            if key not in sim_filter:
                simulation_config.append((key, self._parse_value(value)))

        return simulation_config

    def _extract_config(self, config):
        defines = self._extract_defines(config)
        key_filter = [ConfigKeys.DEFINES]

        new_config = []

        # Extract internals
        key_filter.append(ConfigKeys.INTERNALS)
        config_dir, internal_config = self._extract_internals(config)
        new_config += internal_config

        # Extract simulation
        key_filter.append(ConfigKeys.SIMULATION)
        new_config += self._extract_simulation(config)

        # Unrecognized keys
        for key, value in config.items():
            if key not in key_filter:
                self._failed_keys[key] = value

        return defines, config_dir, new_config

    def _build_config_content(self, config):
        self._failed_keys = {}
        defines, config_dir, config_list = self._extract_config(config)

        config_parser = ConfigParser()
        init_user_config_parser(config_parser)
        config_content = ConfigContent(None)
        config_content.setParser(config_parser)

        # Insert defines
        for key, value in defines.items():
            config_content.add_define(key, value)

        # Insert key values
        if not os.path.exists(config_dir):
            raise IOError(f"The configuration directory: {config_dir} does not exist")

        path_elm = config_content.create_path_elm(config_dir)

        def add_key_value(key, value):
            return config_parser.add_key_value(
                config_content, key, StringList([key] + value), path_elm=path_elm
            )

        for key, value in config_list:
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list):
                raise ValueError(f"Expected value to be str or list, was {type(value)}")

            ok = add_key_value(key, value)
            if not ok:
                self._failed_keys[key] = value

        config_parser.validate(config_content)
        self._errors = list(config_content.getErrors())

        return config_content

    def preferred_num_cpu(self) -> int:
        return int(self.substitution_list.get(f"<{ConfigKeys.NUM_CPU}>", 1))

    def preferred_job_fmt(self) -> str:
        in_config = self.model_config.jobname_format_string
        if in_config is None:
            return "JOB%d"
        else:
            return in_config

    @property
    def errors(self) -> List[str]:
        return self._errors

    @property
    def failed_keys(self):
        return self._failed_keys

    @property
    def ert_templates(self):
        return self._templates

    def __eq__(self, other):
        return (
            other is not None
            and isinstance(other, ResConfig)
            and self.substitution_list == other.substitution_list
            and self.installed_jobs == other.installed_jobs
            and self.env_vars == other.env_vars
            and self.random_seed == other.random_seed
            and self.analysis_config == other.analysis_config
            and self.ert_workflow_list == other.ert_workflow_list
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
            f"Installed jobs: {self.installed_jobs},\n"
            f"EnvironmentVarlist: {self.env_vars},\n"
            f"RandomSeed: {self.random_seed},\n"
            f"Num CPUs: {self.preferred_num_cpu()},\n"
            f"AnalysisConfig: {self.analysis_config},\n"
            f"ErtWorkflowList: {self.ert_workflow_list},\n"
            f"ErtTemplates: {self.ert_templates},\n"
            f"EnsembleConfig: {self.ensemble_config},\n"
            f"ModelConfig: {self.model_config},\n"
            f"QueueConfig: {self.queue_config}"
        )
