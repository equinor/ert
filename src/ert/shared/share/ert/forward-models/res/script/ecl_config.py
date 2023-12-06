import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional

import yaml


def re_getenv(match_obj):
    match_str = match_obj.group(1)
    variable = match_str[1:]
    return os.getenv(variable, default=match_str)


def _replace_env(env: dict) -> dict:
    """Small utility function will take a dict as input, and create a new
    dictionary where all $VARIABLE in the values has been replaced with
    getenv("VARIABLE"). Variables which are not recognized are left
    unchanged."""
    new_env = {}
    for key, value in env.items():
        new_env[key] = re.sub(r"(\$[A-Z0-9_]+)", re_getenv, value)

    return new_env


class Keys:
    default_version: str = "default_version"
    default: str = "default"
    versions: str = "versions"
    env: str = "env"
    mpi: str = "mpi"
    mpirun: str = "mpirun"
    executable: str = "executable"
    scalar: str = "scalar"


class EclConfig:
    """Represent the eclipse configuration at a site.

    The EclConfig class internalizes information of where the various eclipse
    programs are installed on site, and which environment variables need to be
    set before the simulator starts. The class is based on parsing a yaml
    formatted configuration file, the source distribution contains commented
    example file.

    """

    def __init__(self, config_file: str, simulator_name: str = "not_set"):
        with open(config_file, encoding="utf-8") as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Failed parse: {config_file} as yaml") from e

        self._config: dict = config
        self._config_file: str = os.path.abspath(config_file)
        self.simulator_name: str = simulator_name

    def __contains__(self, version: str) -> bool:
        if version in self._config[Keys.versions]:
            return True

        return self.default_version is not None and version in [None, Keys.default]

    def get_eclrun_env(self) -> Optional[Dict[str, str]]:
        if "eclrun_env" in self._config:
            return self._config["eclrun_env"].copy()
        return None

    @property
    def default_version(self) -> Optional[str]:
        return self._config.get(Keys.default_version)

    def _get_version(self, version_arg: Optional[str]) -> str:
        if version_arg in [None, Keys.default]:
            version = self.default_version
        else:
            version = version_arg

        if version is None:
            raise ValueError(
                "The default version has not been "
                f"set in the config file:{self._config_file}"
            )

        return version

    def _get_env(self, version: str, exe_type: str) -> Dict[str, str]:
        env: Dict[str, str] = {}
        env.update(self._config.get(Keys.env, {}))

        mpi_sim: Dict[str, Any] = self._config[Keys.versions][
            self._get_version(version)
        ][exe_type]
        env.update(mpi_sim.get(Keys.env, {}))

        return _replace_env(env)


class Ecl100Config(EclConfig):
    DEFAULT_CONFIG_FILE: str = os.path.join(
        os.path.dirname(__file__), "ecl100_config.yml"
    )

    def __init__(self):
        config_file = os.getenv("ECL100_SITE_CONFIG", default=self.DEFAULT_CONFIG_FILE)
        super().__init__(config_file, simulator_name="eclipse")


class Ecl300Config(EclConfig):
    DEFAULT_CONFIG_FILE: str = os.path.join(
        os.path.dirname(__file__), "ecl300_config.yml"
    )

    def __init__(self):
        config_file = os.getenv("ECL300_SITE_CONFIG", default=self.DEFAULT_CONFIG_FILE)
        super().__init__(config_file, simulator_name="e300")


class FlowConfig(EclConfig):
    DEFAULT_CONFIG_FILE: str = os.path.join(
        os.path.dirname(__file__), "flow_config.yml"
    )

    def __init__(self):
        config_file = os.getenv("FLOW_SITE_CONFIG", default=self.DEFAULT_CONFIG_FILE)
        super().__init__(config_file, simulator_name="flow")


class EclrunConfig:
    """This class contains configurations for using the eclrun binary
    for running eclipse. It uses the configurations classes above to
    get the configuration in the ECLX00_SITE_CONFIG files.
    """

    def __init__(self, config: EclConfig, version: str):
        self.simulator_name: str = config.simulator_name
        self.run_env: Optional[Dict[str, str]] = self._get_run_env(
            config.get_eclrun_env()
        )
        self.version: str = version

    def _get_run_env(
        self, eclrun_env: Optional[Dict[str, str]]
    ) -> Optional[Dict[str, str]]:
        if eclrun_env is None:
            return None

        env: dict = os.environ.copy()
        if "PATH" in eclrun_env:
            env["PATH"] = eclrun_env["PATH"] + os.pathsep + env["PATH"]
            eclrun_env.pop("PATH")

        for key, value in eclrun_env.copy().items():
            if value is None:
                if key in env:
                    env.pop(key)
                eclrun_env.pop(key)

        env.update(eclrun_env)
        return env

    def _get_available_eclrun_versions(self) -> List[str]:
        try:
            return (
                subprocess.check_output(
                    ["eclrun", "--report-versions", self.simulator_name],
                    env=self.run_env,
                )
                .decode("utf-8")
                .strip()
                .split(" ")
            )
        except subprocess.CalledProcessError:
            return []

    def can_use_eclrun(self) -> bool:
        if self.run_env is None:
            return False

        ecl_run_versions = self._get_available_eclrun_versions()
        if self.version not in ecl_run_versions:
            return False

        return True
