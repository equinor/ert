import os
import os.path
import yaml
import shutil


class RMSConfig(object):
    DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "rms_config.yml")

    def __init__(self):
        config_file = os.getenv("RMS_SITE_CONFIG", default=self.DEFAULT_CONFIG_FILE)
        with open(config_file) as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError:
                raise ValueError("Failed to parse: {} as yaml".format(config_file))

        self._config = config

    @property
    def executable(self):
        exe = self._config["executable"]
        if not os.access(exe, os.X_OK):
            raise OSError("The executable: {} can not run".format(exe))

        return exe

    @property
    def wrapper(self):
        exe = self._config.get("wrapper", None)
        if exe is not None and shutil.which(exe) is None:
            raise OSError("The executable: {} is not found".format(exe))
        return exe

    @property
    def threads(self):
        return self._config.get("threads")

    def env(self, version):
        env_versions = self._config.get("env", {})
        return env_versions.get(version, {})
