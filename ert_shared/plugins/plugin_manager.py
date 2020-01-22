import functools
import logging
import os
import shutil
import sys
import tempfile
from itertools import chain

import pluggy

_PLUGIN_NAMESPACE = "ert"

hook_implementation = pluggy.HookimplMarker(_PLUGIN_NAMESPACE)
hook_specification = pluggy.HookspecMarker(_PLUGIN_NAMESPACE)

# Imports below hook_implementation and hook_specification to avoid circular imports
import ert_shared.plugins.hook_specifications
import ert_shared.hook_implementations


def python3only(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        if sys.version_info.major >= 3:
            return func(*args, **kwargs)

    return wrapper_decorator


class ErtPluginManager(pluggy.PluginManager):
    @python3only
    def __init__(self, plugins=None):
        super().__init__(_PLUGIN_NAMESPACE)
        self.add_hookspecs(ert_shared.plugins.hook_specifications)
        if plugins is None:
            self.register(ert_shared.hook_implementations)
            self.load_setuptools_entrypoints(_PLUGIN_NAMESPACE)
        else:
            for plugin in plugins:
                self.register(plugin)
        logging.debug(str(self))

    @python3only
    def __str__(self):
        self_str = "ERT Plugin manager:\n"
        for plugin in self.get_plugins():
            self_str += "\t" + self.get_name(plugin) + "\n"
            for hook_caller in self.get_hookcallers(plugin):
                self_str += "\t\t" + str(hook_caller) + "\n"
        return self_str

    @python3only
    def get_help_links(self):
        return ErtPluginManager._merge_dicts(self.hook.help_links())

    @staticmethod
    def _evaluate_config_hook(hook, config_name):
        response = hook()

        if response is None:
            logging.debug("Got no {} config path from any plugins".format(config_name))
            return None

        logging.debug(
            "Got {} config path from {} ({})".format(
                config_name,
                response.plugin_metadata.plugin_name,
                response.plugin_metadata.function_name,
            )
        )
        return response.data

    @python3only
    def get_ecl100_config_path(self):
        return ErtPluginManager._evaluate_config_hook(
            hook=self.hook.ecl100_config_path, config_name="ecl100"
        )

    @python3only
    def get_ecl300_config_path(self):
        return ErtPluginManager._evaluate_config_hook(
            hook=self.hook.ecl300_config_path, config_name="ecl300"
        )

    @python3only
    def get_flow_config_path(self):
        return ErtPluginManager._evaluate_config_hook(
            hook=self.hook.flow_config_path, config_name="flow"
        )

    @python3only
    def get_rms_config_path(self):
        return ErtPluginManager._evaluate_config_hook(
            hook=self.hook.rms_config_path, config_name="rms"
        )

    @python3only
    def _site_config_lines(self):
        try:
            plugin_responses = self.hook.site_config_lines()
        except AttributeError:
            return []
        plugin_site_config_lines = [
            [
                "-- Content below originated from {} ({})".format(
                    plugin_response.plugin_metadata.plugin_name,
                    plugin_response.plugin_metadata.function_name,
                )
            ]
            + plugin_response.data
            for plugin_response in plugin_responses
        ]
        return list(chain.from_iterable(reversed(plugin_site_config_lines)))

    @python3only
    def get_site_config_content(self):
        site_config_lines = self._site_config_lines()

        config_env_vars = {
            "ECL100_SITE_CONFIG": self.get_ecl100_config_path(),
            "ECL300_SITE_CONFIG": self.get_ecl300_config_path(),
            "FLOW_SITE_CONFIG": self.get_flow_config_path(),
            "RMS_SITE_CONFIG": self.get_rms_config_path(),
        }
        config_lines = [
            "SETENV {} {}".format(env_var, env_value)
            for env_var, env_value in config_env_vars.items()
            if env_value is not None
        ]
        site_config_lines.extend(config_lines + [""])

        install_job_lines = [
            "INSTALL_JOB {} {}".format(job_name, job_path)
            for job_name, job_path in self.get_installable_jobs().items()
        ]

        site_config_lines.extend(install_job_lines + [""])

        install_workflow_job_lines = [
            "LOAD_WORKFLOW_JOB {}".format(job_path)
            for job_name, job_path in self.get_installable_workflow_jobs().items()
        ]

        site_config_lines.extend(install_workflow_job_lines + [""])

        return "\n".join(site_config_lines) + "\n"

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
                    "Overwriting {} from {}({}) with data from {}({})".format(
                        ck,
                        merged_dict[ck][1].plugin_name,
                        merged_dict[ck][1].function_name,
                        d.plugin_metadata.plugin_name,
                        d.plugin_metadata.function_name,
                    )
                )
            merged_dict.update(ErtPluginManager._add_plugin_info_to_dict(d.data, d))

        if include_plugin_data:
            return merged_dict
        return {k: v[0] for k, v in merged_dict.items()}

    @python3only
    def get_installable_jobs(self):
        return ErtPluginManager._merge_dicts(self.hook.installable_jobs())

    @python3only
    def get_installable_workflow_jobs(self):
        return ErtPluginManager._merge_dicts(self.hook.installable_workflow_jobs())


class ErtPluginContext:
    @python3only
    def __init__(self, plugins=None):
        self.plugin_manager = ErtPluginManager(plugins=plugins)
        self.tmp_dir = None
        self.tmp_site_config_filename = None

    @python3only
    def _create_site_config(self):
        site_config_content = self.plugin_manager.get_site_config_content()
        tmp_site_config_filename = None
        if site_config_content is not None:
            logging.debug("Creating temporary site-config")
            tmp_site_config_filename = os.path.join(self.tmp_dir, "site-config")
            with open(tmp_site_config_filename, "w") as fh:
                fh.write(site_config_content)
            logging.debug(
                "Temporary site-config created: {}".format(tmp_site_config_filename)
            )
        return tmp_site_config_filename

    @python3only
    def __enter__(self):
        logging.debug("Creating temporary directory for site-config")
        self.tmp_dir = tempfile.mkdtemp()
        logging.debug("Temporary directory created: {}".format(self.tmp_dir))
        self.tmp_site_config_filename = self._create_site_config()
        env = {
            "ERT_SITE_CONFIG": self.tmp_site_config_filename,
        }
        self._setup_temp_environment_if_not_already_set(env)
        return self

    @python3only
    def _setup_temp_environment_if_not_already_set(self, env):
        self.backup_env = os.environ.copy()
        self.env = env

        for name, value in env.items():
            if self.backup_env.get(name) is None:
                if value is not None:
                    logging.debug(
                        "Setting environment variable {}={}".format(name, value)
                    )
                    os.environ[name] = value
            else:
                logging.debug(
                    "Environment variable already set {}={}, leaving it as is".format(
                        name, self.backup_env.get(name)
                    )
                )

    @python3only
    def _reset_environment(self):
        for name, value in self.env.items():
            if self.backup_env.get(name) is None and name in os.environ:
                logging.debug("Resetting environment variable {}".format(name))
                del os.environ[name]

    @python3only
    def __exit__(self, *args):
        self._reset_environment()
        logging.debug("Deleting temporary directory for site-config")
        shutil.rmtree(self.tmp_dir)
