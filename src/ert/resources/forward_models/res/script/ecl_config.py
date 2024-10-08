import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
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


class Simulator:
    """Small 'struct' with the config information for one simulator."""

    def __init__(
        self,
        version: str,
        executable: str,
        env: Dict[str, str],
        mpirun: Optional[str] = None,
    ):
        self.version: str = version
        if not os.access(executable, os.X_OK):
            raise OSError(f"The executable: '{executable}' can not be executed by user")

        self.executable: str = executable
        self.env: Dict[str, str] = env
        self.mpirun: Optional[str] = mpirun
        self.name: str = "simulator"

        if mpirun is not None and not os.access(mpirun, os.X_OK):
            raise OSError(f"The mpirun binary: '{mpirun}' is not executable by user")

    def __repr__(self) -> str:
        mpistring: str = ""
        if self.mpirun:
            mpistring = " MPI"
        return (
            f"{self.name}(version={self.version}, "
            f"executable={self.executable}{mpistring})"
        )


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

    def _get_sim(self, version: Optional[str], exe_type: str) -> Simulator:
        version = self._get_version(version)
        binaries: Dict[str, str] = self._config[Keys.versions][version][exe_type]
        mpirun = binaries[Keys.mpirun] if exe_type == Keys.mpi else None
        return Simulator(
            version,
            binaries[Keys.executable],
            self._get_env(version, exe_type),
            mpirun=mpirun,
        )

    def sim(self, version: Optional[str] = None) -> Simulator:
        """Will return an object describing the simulator.

        Available attributes are 'executable' and 'env'. Observe that the
        executable path is validated when you instantiate the Simulator object;
        so if the executable key in the config file points to non-existing file
        you will not get the error before this point.
        """
        return self._get_sim(version, Keys.scalar)

    def mpi_sim(self, version: Optional[str] = None) -> Simulator:
        """MPI version of method sim()"""
        return self._get_sim(version, Keys.mpi)

    def simulators(self, strict: bool = True) -> List[Simulator]:
        simulators = []
        for version in self._config[Keys.versions]:
            for exe_type in self._config[Keys.versions][version]:
                if strict:
                    sim = self._get_sim(version, exe_type)
                else:
                    try:
                        sim = self._get_sim(version, exe_type)
                    except OSError:
                        sys.stderr.write(
                            "Failed to create simulator object for: "
                            f"version:{version} {exe_type}\n"
                        )
                        sim = None

                if sim:
                    simulators.append(sim)
        return simulators


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
        if not Path(config_file).exists():
            config_file = self.init_flow_config()
        super().__init__(config_file, simulator_name="flow")

    @staticmethod
    def init_flow_config() -> str:
        binary_path = shutil.which("flow")
        if binary_path is None:
            raise FileNotFoundError(
                "Could not find flow executable!\n"
                " Requires flow to be installed in $PATH"
            )

        conf = {
            "default_version": "default",
            "versions": {"default": {"scalar": {"executable": binary_path}}},
        }
        flow_config_yml = Path("flow_config.yml")
        flow_config_yml.write_text(yaml.dump(conf), encoding="utf-8")
        return str(flow_config_yml.absolute())


class EclrunConfig:
    """This class contains configurations for using the new eclrun binary
    for running eclipse. It uses the old configurations classes above to
    get the configuration in the ECLX00_SITE_CONFIG files.
    """

    def __init__(self, config: EclConfig, version: str):
        self.simulator_name: str = config.simulator_name
        self.run_env: Optional[Dict[str, str]] = self._get_run_env(
            config.get_eclrun_env()
        )
        self.version: str = version

    @staticmethod
    def _get_run_env(eclrun_env: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
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

        return self.version in self._get_available_eclrun_versions()
