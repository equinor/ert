import os
import shutil
import tempfile

from everest.plugins.hook_manager import EverestPluginManager


class PluginSiteConfigEnv:
    """
    Allows plugin configuration of site config file.
    """

    def __init__(self):
        self.pm = EverestPluginManager()
        self.backup_env = os.environ.copy()
        self.tmp_dir = None

    def _config_env_vars(self):
        config_env_vars = [
            ("ECL100_SITE_CONFIG", self.pm.hook.ecl100_config_path()),
            ("ECL300_SITE_CONFIG", self.pm.hook.ecl300_config_path()),
            ("FLOW_SITE_CONFIG", self.pm.hook.flow_config_path()),
        ]
        config_lines = [
            "SETENV {} {}".format(env_var, env_value.data)
            for env_var, env_value in config_env_vars
            if env_value is not None
        ]

        return [*config_lines, ""]

    def _get_temp_site_config_path(self):
        self.tmp_dir = tempfile.mkdtemp()
        return os.path.join(self.tmp_dir, "site-config")

    def _install_workflow_job_lines(self):
        response = self.pm.hook.installable_workflow_jobs()
        if response:
            job_paths = []
            for item in reversed(response):
                job_paths += item.data.values()
            return ["LOAD_WORKFLOW_JOB {}".format(path) for path in job_paths] + [""]
        return []

    def _site_config_content(self):
        response = self.pm.hook.site_config_lines()
        if response:
            lines = []
            for item in reversed(response):
                lines += item.data
            return lines
        return None

    def _get_site_config_content(self):
        plugin_content = self._site_config_content()
        if plugin_content:
            site_config_lines = self.pm.hook.default_site_config_lines()
            site_config_lines.extend(plugin_content)
            site_config_lines.extend(self._config_env_vars())
            site_config_lines.extend(self.pm.hook.install_job_directories() or [])
            site_config_lines.extend(self._install_workflow_job_lines())

            return "\n".join(site_config_lines) + "\n"
        return None

    @staticmethod
    def _is_site_config_env_set():
        return os.environ.get("ERT_SITE_CONFIG", None) is not None

    @staticmethod
    def _write_tmp_site_config_file(path, content):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)

    def __enter__(self):
        if not self._is_site_config_env_set():
            site_config_content = self._get_site_config_content()
            if site_config_content is not None:
                tmp_site_conf_path = self._get_temp_site_config_path()
                self._write_tmp_site_config_file(
                    tmp_site_conf_path, site_config_content
                )
                os.environ["ERT_SITE_CONFIG"] = tmp_site_conf_path

    def __exit__(self, *args):
        if (
            self.backup_env.get("ERT_SITE_CONFIG", None) is None
            and self._is_site_config_env_set()
        ):
            del os.environ["ERT_SITE_CONFIG"]

        if self.tmp_dir is not None:
            shutil.rmtree(self.tmp_dir)
