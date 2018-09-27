import os
import yaml
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
    for key,value in env.items():
        new_env[key] = re.sub(r"(\$[A-Z0-9_]+)", re_getenv, value)

    return new_env


class Simulator(object):
    """Small 'struct' with the config information for one simulator.
    """
    def __init__(self, name, version, executable, env, mpirun = None):
        self.name = name
        self.version = version
        if not os.access( executable , os.X_OK ):
            raise OSError("The executable: '{}' can not be executed by user".format(executable))

        self.executable = executable
        self.env = env
        self.mpirun = mpirun

        if not mpirun is None:
            if not os.access(mpirun, os.X_OK):
                raise OSError("The mpirun argument: '{}' is not executable by user".format(executable))


    def __repr__(self):
        mpistring = ""
        if self.mpirun:
            mpistring = " MPI"
        return "{}(version={}, executable={}{})".format(self.name, self.version, self.executable, mpistring)



class EclConfig(object):
    """Represent the eclipse configuration at a site.

    The EclConfig class internalizes information of where the various eclipse
    programs are installed on site, and which environment variables need to be
    set before the simulator starts. The class is based on parsing a yaml
    formatted configuration file, the source distribution contains commented
    example file.

    """
    DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "ecl_config.yml")

    def __init__(self):
        config_file = os.getenv("ECL_SITE_CONFIG", default = self.DEFAULT_CONFIG_FILE)
        with open(config_file) as f:
            try:
                config = yaml.load(f)
            except:
                raise ValueError("Failed parse: {} as yaml".format(config_file))

        self._config = config


    def _get_env(self, sim, version, exe_type):
        env = {}
        env.update( self._config.get("env", {} ))

        mpi_sim = self._config["simulators"][sim][version][exe_type]
        env.update( mpi_sim.get("env", {}))
        return _replace_env(env)


    def _get_sim(self, sim, version, exe_type):
        d = self._config["simulators"][sim][version][exe_type]
        if exe_type == "mpi":
            mpirun = d["mpirun"]
        else:
            mpirun = None
        return Simulator(sim, version, d["executable"], self._get_env(sim, version, exe_type), mpirun = mpirun)


    def sim(self, sim, version):
        """Will return a small struct describing the simulator.

        The struct has attributes 'executable' and 'env'. Observe that the
        executable path is validated when you instantiate the Simulator object;
        so if the executable key in the config file points to non-existing file
        you will not get the error before this point.
        """
        return self._get_sim(sim, version, "scalar")

    def mpi_sim(self, sim, version):
        """MPI version of method sim()."""
        return self._get_sim(sim, version, "mpi")


    def simulators(self, strict = True):
        simulators = []
        for name,versions in self._config["simulators"].items():
            for version,exe_types in versions.items():
                for exe_type in exe_types.keys():
                    if strict:
                        sim = self._get_sim(name, version, exe_type)
                    else:
                        try:
                            sim = self._get_sim(name, version, exe_type)
                        except Exception:
                            sys.stderr.write("Failed to create simulator object for:{name} version:{version} {exe_type}\n".format(name=name,
                                                                                                                                  version=version,
                                                                                                                                  exe_type=exe_type))
                            sim = None

                    if sim:
                        simulators.append(sim)
        return simulators
