from argparse import ArgumentParser
from .ecl_run import EclRun
from res.fm.ecl.ecl_config import EclrunConfig


def run(config, argv):
    parser = ArgumentParser()
    parser.add_argument("ecl_case")
    parser.add_argument("-v", "--version", dest="version", type=str)
    parser.add_argument("-n", "--num-cpu", dest="num_cpu", type=int, default=1)
    parser.add_argument(
        "-i", "--ignore-errors", dest="ignore_errors", action="store_true"
    )

    options = parser.parse_args(argv)

    eclrun_config = EclrunConfig(config, options.version)

    if options.num_cpu > 1:
        sim = config.mpi_sim(version=options.version)
    else:
        sim = config.sim(version=options.version)

    run = EclRun(
        options.ecl_case,
        sim,
        num_cpu=options.num_cpu,
        check_status=not options.ignore_errors,
    )
    run.runEclipse()
