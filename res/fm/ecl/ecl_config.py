import os
import yaml
import subprocess
import sys
import re


def re_getenv(match_obj):
    match_str = match_obj.group(1)
    variable = match_str[1:]
    return os.getenv(variable, default=match_str)


# Small utility function will take a dict as input, and create a new dictionary
# where all $VARIABLE in the values has been replaced with getenv("VARIABLE").
# Variables which are not recognized are left unchanged.


def _replace_env(env):
    new_env = {}
    for key, value in env.items():
        new_env[key] = re.sub(r"(\$[A-Z0-9_]+)", re_getenv, value)

    return new_env


class Keys(object):
    default_version = "default_version"
    default = "default"
    versions = "versions"
    env = "env"
    mpi = "mpi"
    mpirun = "mpirun"
    executable = "executable"
    scalar = "scalar"


class Simulator(object):
    """Small 'struct' with the config information for one simulator."""

    def __init__(self, version, executable, env, mpirun=None):
        self.version = version
        if not os.access(executable, os.X_OK):
            raise OSError(
                "The executable: '{}' can not be executed by user".format(executable)
            )

        self.executable = executable
        self.env = env
        self.mpirun = mpirun
        self.name = "simulator"

        if not mpirun is None:
            if not os.access(mpirun, os.X_OK):
                raise OSError(
                    "The mpirun argument: '{}' is not executable by user".format(
                        executable
                    )
                )

    def __repr__(self):
        mpistring = ""
        if self.mpirun:
            mpistring = " MPI"
        return "{}(version={}, executable={}{})".format(
            self.name, self.version, self.executable, mpistring
        )


class EclConfig(object):
    """Represent the eclipse configuration at a site.

    The EclConfig class internalizes information of where the various eclipse
    programs are installed on site, and which environment variables need to be
    set before the simulator starts. The class is based on parsing a yaml
    formatted configuration file, the source distribution contains commented
    example file.

    """

    def __init__(self, config_file, simulator_name="not_set"):
        with open(config_file) as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError:
                raise ValueError("Failed parse: {} as yaml".format(config_file))

        self._config = config
        self._config_file = os.path.abspath(config_file)
        self.simulator_name = simulator_name

    def __contains__(self, version):
        if version in self._config[Keys.versions]:
            return True

        return self.default_version is not None and version in [None, Keys.default]

    def get_eclrun_env(self):
        if "eclrun_env" in self._config:
            return self._config["eclrun_env"].copy()
        return None

    @property
    def default_version(self):
        return self._config.get(Keys.default_version)

    def _get_version(self, version_arg):
        if version_arg in [None, Keys.default]:
            version = self.default_version
        else:
            version = version_arg

        if version is None:
            raise ValueError(
                "The default version has not not been set in the config file:{}".format(
                    self._config_file
                )
            )

        return version

    def _get_env(self, version, exe_type):
        env = {}
        env.update(self._config.get(Keys.env, {}))

        version = self._get_version(version)
        mpi_sim = self._config[Keys.versions][version][exe_type]
        env.update(mpi_sim.get(Keys.env, {}))
        return _replace_env(env)

    def _get_sim(self, version, exe_type):
        version = self._get_version(version)
        d = self._config[Keys.versions][version][exe_type]
        if exe_type == Keys.mpi:
            mpirun = d[Keys.mpirun]
        else:
            mpirun = None
        return Simulator(
            version, d[Keys.executable], self._get_env(version, exe_type), mpirun=mpirun
        )

    def sim(self, version=None):
        """Will return a small struct describing the simulator.

        The struct has attributes 'executable' and 'env'. Observe that the
        executable path is validated when you instantiate the Simulator object;
        so if the executable key in the config file points to non-existing file
        you will not get the error before this point.
        """
        return self._get_sim(version, Keys.scalar)

    def mpi_sim(self, version=None):
        """MPI version of method sim()."""
        return self._get_sim(version, Keys.mpi)

    def simulators(self, strict=True):
        simulators = []
        for version in self._config[Keys.versions].keys():
            for exe_type in self._config[Keys.versions][version].keys():
                if strict:
                    sim = self._get_sim(version, exe_type)
                else:
                    try:
                        sim = self._get_sim(version, exe_type)
                    # This exception should be more specific after resolving
                    # https://github.com/equinor/ert/issues/1955
                    except Exception:
                        sys.stderr.write(
                            "Failed to create simulator object for: version:{version} {exe_type}\n".format(
                                version=version, exe_type=exe_type
                            )
                        )
                        sim = None

                if sim:
                    simulators.append(sim)
        return simulators


class Ecl100Config(EclConfig):

    DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "ecl100_config.yml")

    def __init__(self):
        config_file = os.getenv("ECL100_SITE_CONFIG", default=self.DEFAULT_CONFIG_FILE)
        super(Ecl100Config, self).__init__(config_file, simulator_name="eclipse")


class Ecl300Config(EclConfig):

    DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "ecl300_config.yml")

    def __init__(self):
        config_file = os.getenv("ECL300_SITE_CONFIG", default=self.DEFAULT_CONFIG_FILE)
        super(Ecl300Config, self).__init__(config_file, simulator_name="e300")


class FlowConfig(EclConfig):

    DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "flow_config.yml")

    def __init__(self):
        config_file = os.getenv("FLOW_SITE_CONFIG", default=self.DEFAULT_CONFIG_FILE)
        super(FlowConfig, self).__init__(config_file, simulator_name="flow")


class EclrunConfig:
    """This class contains configurations for using the new eclrun binary
    for running eclipse. It uses the old configurations classes above to
    get the configuration in the ECLX00_SITE_CONFIG files.
    """

    def __init__(self, config, version):
        self.simulator_name = config.simulator_name
        self.run_env = self._get_run_env(config.get_eclrun_env())
        self.version = version

    def _get_run_env(self, eclrun_env):
        if eclrun_env is None:
            return None

        env = os.environ.copy()
        if "PATH" in eclrun_env:
            env["PATH"] = eclrun_env["PATH"] + os.pathsep + env["PATH"]
            eclrun_env.pop("PATH")

        for k, v in eclrun_env.copy().items():
            if v is None:
                if k in env:
                    env.pop(k)
                eclrun_env.pop(k)

        env.update(eclrun_env)
        return env

    def _get_available_eclrun_versions(self):
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

    def can_use_eclrun(self):
        if self.run_env is None:
            return False

        ecl_run_versions = self._get_available_eclrun_versions()
        if self.version not in ecl_run_versions:
            return False

        return True
