import logging
import os
import shutil
import tempfile
from itertools import chain

import pluggy

from ert.shared.plugins.workflow_config import WorkflowConfigs

_PLUGIN_NAMESPACE = "ert"

hook_implementation = pluggy.HookimplMarker(_PLUGIN_NAMESPACE)
hook_specification = pluggy.HookspecMarker(_PLUGIN_NAMESPACE)

import ert.shared.hook_implementations  # noqa

# Imports below hook_implementation and hook_specification to avoid circular imports
import ert.shared.plugins.hook_specifications  # noqa


# pylint: disable=no-member
# (pylint does not know what the plugins offer)
class ErtPluginManager(pluggy.PluginManager):
    def __init__(self, plugins=None):
        super().__init__(_PLUGIN_NAMESPACE)
        self.add_hookspecs(ert.shared.plugins.hook_specifications)
        if plugins is None:
            self.register(ert.shared.hook_implementations)
            self.load_setuptools_entrypoints(_PLUGIN_NAMESPACE)
        else:
            for plugin in plugins:
                self.register(plugin)
        logging.debug(str(self))

    def __str__(self):
        self_str = "ERT Plugin manager:\n"
        for plugin in self.get_plugins():
            self_str += "\t" + self.get_name(plugin) + "\n"
            for hook_caller in self.get_hookcallers(plugin):
                self_str += "\t\t" + str(hook_caller) + "\n"
        return self_str

    def get_help_links(self):
        return ErtPluginManager._merge_dicts(self.hook.help_links())

    @staticmethod
    def _evaluate_config_hook(hook, config_name):
        response = hook()

        if response is None:
            logging.debug(f"Got no {config_name} config path from any plugins")
            return None

        logging.debug(
            f"Got {config_name} config path from "
            f"{response.plugin_metadata.plugin_name} "
            f"({response.plugin_metadata.function_name,})"
        )
        return response.data

    @staticmethod
    def _evaluate_job_doc_hook(hook, job_name):

        response = hook(job_name=job_name)

        if response is None:
            logging.debug(f"Got no documentation for {job_name} from any plugins")
            return {}

        return response.data

    def get_ecl100_config_path(self):
        return ErtPluginManager._evaluate_config_hook(
            hook=self.hook.ecl100_config_path, config_name="ecl100"
        )

    def get_ecl300_config_path(self):
        return ErtPluginManager._evaluate_config_hook(
            hook=self.hook.ecl300_config_path, config_name="ecl300"
        )

    def get_flow_config_path(self):
        return ErtPluginManager._evaluate_config_hook(
            hook=self.hook.flow_config_path, config_name="flow"
        )

    def get_rms_config_path(self):
        return ErtPluginManager._evaluate_config_hook(
            hook=self.hook.rms_config_path, config_name="rms"
        )

    def _site_config_lines(self):
        try:
            plugin_responses = self.hook.site_config_lines()
        except AttributeError:
            return []
        plugin_site_config_lines = [
            [
                "-- Content below originated from "
                f"{plugin_response.plugin_metadata.plugin_name} "
                f"({plugin_response.plugin_metadata.function_name})"
            ]
            + plugin_response.data
            for plugin_response in plugin_responses
        ]
        return list(chain.from_iterable(reversed(plugin_site_config_lines)))

    def get_installable_workflow_jobs(self):
        config_workflow_jobs = self._get_config_workflow_jobs()
        hooked_workflow_jobs = self.get_ertscript_workflows().get_workflows()
        installable_workflow_jobs = self._merge_internal_jobs(
            config_workflow_jobs, hooked_workflow_jobs
        )
        return installable_workflow_jobs

    def get_site_config_content(self):
        site_config_lines = self._site_config_lines()

        config_env_vars = {
            "ECL100_SITE_CONFIG": self.get_ecl100_config_path(),
            "ECL300_SITE_CONFIG": self.get_ecl300_config_path(),
            "FLOW_SITE_CONFIG": self.get_flow_config_path(),
            "RMS_SITE_CONFIG": self.get_rms_config_path(),
        }
        config_lines = [
            f"SETENV {env_var} {env_value}"
            for env_var, env_value in config_env_vars.items()
            if env_value is not None
        ]
        site_config_lines.extend(config_lines + [""])

        install_job_lines = [
            f"INSTALL_JOB {job_name} {job_path}"
            for job_name, job_path in self.get_installable_jobs().items()
        ]

        site_config_lines.extend(install_job_lines + [""])

        installable_workflow_jobs = self.get_installable_workflow_jobs()

        install_workflow_job_lines = [
            f"LOAD_WORKFLOW_JOB {job_path}"
            for job_name, job_path in installable_workflow_jobs.items()
        ]
        site_config_lines.extend(install_workflow_job_lines + [""])

        return "\n".join(site_config_lines) + "\n"

    @staticmethod
    def _merge_internal_jobs(config_jobs, hooked_jobs):
        conflicting_keys = set(config_jobs.keys()) & set(hooked_jobs.keys())
        for ck in conflicting_keys:
            logging.info(
                f"Duplicate key: {ck} in workflow hook implementations, "
                f"config path 1: {config_jobs[ck]}, "
                f"config path 2: {hooked_jobs[ck]}"
            )
        merged_jobs = config_jobs.copy()
        merged_jobs.update(hooked_jobs)
        return merged_jobs

    @staticmethod
    def _add_plugin_info_to_dict(d, plugin_response):
        return {k: (v, plugin_response.plugin_metadata) for k, v in d.items()}

    @staticmethod
    def _merge_dicts(list_of_dicts, include_plugin_data=False):
        list_of_dicts.reverse()
        merged_dict = {}
        for d in list_of_dicts:
            conflicting_keys = set(merged_dict.keys()) & set(d.data.keys())
            for ck in conflicting_keys:
                logging.info(
                    f"Overwriting {ck} from "
                    f"{merged_dict[ck][1].plugin_name}"
                    f"({merged_dict[ck][1].function_name}) "
                    f"with data from {d.plugin_metadata.plugin_name}"
                    f"({d.plugin_metadata.function_name})"
                )
            merged_dict.update(ErtPluginManager._add_plugin_info_to_dict(d.data, d))

        if include_plugin_data:
            return merged_dict
        return {k: v[0] for k, v in merged_dict.items()}

    def get_installable_jobs(self):
        return ErtPluginManager._merge_dicts(self.hook.installable_jobs())

    def _get_config_workflow_jobs(self):
        return ErtPluginManager._merge_dicts(self.hook.installable_workflow_jobs())

    def get_documentation_for_jobs(self):
        job_docs = {
            k: {
                "config_file": v[0],
                "source_package": v[1].plugin_name,
                "source_function_name": v[1].function_name,
            }
            for k, v in ErtPluginManager._merge_dicts(
                self.hook.installable_jobs(), include_plugin_data=True
            ).items()
        }
        for k in job_docs.keys():
            job_docs[k].update(
                ErtPluginManager._evaluate_job_doc_hook(self.hook.job_documentation, k)
            )
        return job_docs

    def get_documentation_for_workflows(self):
        workflow_config = self.get_ertscript_workflows()

        job_docs = {
            workflow.name: {
                "description": workflow.description,
                "examples": workflow.examples,
                "config_file": workflow.config_path,
                "parser": workflow.parser,
                "source_package": workflow.source_package,
                "category": workflow.category,
            }
            for workflow in workflow_config._workflows
        }

        return job_docs

    def get_ertscript_workflows(self):
        config = WorkflowConfigs()
        self.hook.legacy_ertscript_workflow(config=config)
        return config

    def add_logging_handle_to_root(self, logger):
        handles = self.hook.add_log_handle_to_root()
        for handle in handles:
            logger.addHandler(handle)


class ErtPluginContext:
    def __init__(self, plugins=None):
        self.plugin_manager = ErtPluginManager(plugins=plugins)
        self.tmp_dir = None
        self.tmp_site_config_filename = None

    def _create_site_config(self):
        site_config_content = self.plugin_manager.get_site_config_content()
        tmp_site_config_filename = None
        if site_config_content is not None:
            logging.debug("Creating temporary site-config")
            tmp_site_config_filename = os.path.join(self.tmp_dir, "site-config")
            with open(tmp_site_config_filename, "w") as fh:
                fh.write(site_config_content)
            logging.debug(f"Temporary site-config created: {tmp_site_config_filename}")
        return tmp_site_config_filename

    def __enter__(self):
        logging.debug("Creating temporary directory for site-config")
        self.tmp_dir = tempfile.mkdtemp()
        logging.debug(f"Temporary directory created: {self.tmp_dir}")
        self.tmp_site_config_filename = self._create_site_config()
        env = {
            "ERT_SITE_CONFIG": self.tmp_site_config_filename,
        }
        self._setup_temp_environment_if_not_already_set(env)
        return self

    def _setup_temp_environment_if_not_already_set(self, env):
        self.backup_env = os.environ.copy()
        self.env = env

        for name, value in env.items():
            if self.backup_env.get(name) is None:
                if value is not None:
                    logging.debug(f"Setting environment variable {name}={value}")
                    os.environ[name] = value
            else:
                logging.debug(
                    f"Environment variable already set "
                    f"{name}={self.backup_env.get(name)}, leaving it as is"
                )

    def _reset_environment(self):
        for name in self.env.keys():
            if self.backup_env.get(name) is None and name in os.environ:
                logging.debug(f"Resetting environment variable {name}")
                del os.environ[name]

    def __exit__(self, *args):
        self._reset_environment()
        logging.debug("Deleting temporary directory for site-config")
        shutil.rmtree(self.tmp_dir)
