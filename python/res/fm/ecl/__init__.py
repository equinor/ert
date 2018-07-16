from .ecl_config import EclConfig
from .ecl_run import EclRun


def simulate(simulator, version, data_file, num_cpu = 1, check = True):
    run_cmd = "run_{}".format(simulator)
    if not check:
        run_cmd += "_nocheck"

    ecl_run = EclRun( [run_cmd, version, data_file, num_cpu])
    ecl_run.runEclipse( )
