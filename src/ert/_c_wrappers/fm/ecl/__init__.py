from .ecl_config import Ecl100Config, Ecl300Config, EclrunConfig, FlowConfig
from .ecl_run import EclRun
from .script import run


# This is very deprecated, should rather use the run() method in the script
# module. That will handle argument parsing and so on.
def simulate(simulator, version, data_file, num_cpu=1, check=True):
    if simulator == "ecl100":
        config = Ecl100Config()
    elif simulator == "flow":
        config = FlowConfig()
    elif simulator == "ecl300":
        config = Ecl300Config()
    else:
        raise Exception(f"No such simulator: {simulator}")

    argv = [data_file, f"--num-cpu={num_cpu}"]

    if version is not None:
        argv.append(f"--version={version}")

    if not check:
        argv.append("--ignore-errors")

    run(config, argv)
