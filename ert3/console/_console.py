import ert3

import argparse
from pathlib import Path
import sys


_ERT3_DESCRIPTION = (
    "ert3 is an ensemble-based tool for uncertainty studies.\n"
    "\nWARNING: the tool is currently in an extremely early stage and we refer "
    "all users to ert for real work!"
)


def _build_argparser():
    parser = argparse.ArgumentParser(description=_ERT3_DESCRIPTION)
    subparsers = parser.add_subparsers(dest="sub_cmd", help="ert3 commands")

    subparsers.add_parser("init", help="Initialize an ERT3 workspace")

    run_parser = subparsers.add_parser("run", help="Run experiment")
    run_parser.add_argument("experiment_name", help="Name of the experiment")

    export_parser = subparsers.add_parser("export", help="Export experiment")
    export_parser.add_argument("experiment_name", help="Name of the experiment")

    sample_parser = subparsers.add_parser(
        "sample", help="Sample parameter distribution"
    )
    sample_parser.add_argument(
        "parameter_group", help="Name of the distribution group in parameters.yml"
    )
    sample_parser.add_argument("sample_name", help="Name of the sampling")
    sample_parser.add_argument(
        "ensemble_size", type=int, help="Size of ensemble of variables"
    )

    return parser


def main():
    parser = _build_argparser()
    args = parser.parse_args()

    # Commands that does not require an ert workspace
    if args.sub_cmd is None:
        parser.print_help()
        return
    if args.sub_cmd == "init":
        ert3.workspace.initialize(Path.cwd())
        return

    # Commands that does requires an ert workspace
    workspace = ert3.workspace.load(Path.cwd())
    if workspace is None:
        sys.exit("Not inside an ERT workspace")
    if args.sub_cmd == "run":
        ert3.engine.run(workspace, args.experiment_name)
        return
    if args.sub_cmd == "export":
        ert3.engine.export(workspace, args.experiment_name)
        return
    if args.sub_cmd == "sample":
        ert3.engine.sample(
            workspace, args.parameter_group, args.sample_name, args.ensemble_size
        )
        return
