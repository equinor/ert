import ert3

import argparse
from pathlib import Path
import sys
import yaml

_ERT3_DESCRIPTION = (
    "ert3 is an ensemble-based tool for uncertainty studies.\n"
    "\nWARNING: the tool is currently in an extremely early stage and we refer "
    "all users to ert for real work!"
)


def _build_record_argparser(subparsers):
    record_parser = subparsers.add_parser("record", help="Record operations")
    sub_record_parsers = record_parser.add_subparsers(
        dest="sub_record_cmd", help="ert3 record operations"
    )
    record_load_parser = sub_record_parsers.add_parser(
        "load", help="Load JSON records from file"
    )
    record_load_parser.add_argument("record_name", help="Name of the resulting record")
    record_load_parser.add_argument(
        "record_file",
        type=argparse.FileType("r"),
        help="Path to resource file",
    )
    sample_parser = sub_record_parsers.add_parser(
        "sample", help="Sample stochastic parameter into a record"
    )
    sample_parser.add_argument(
        "parameter_group", help="Name of the distribution group in parameters.yml"
    )
    sample_parser.add_argument("record_name", help="Name of the resulting record")
    sample_parser.add_argument(
        "ensemble_size", type=int, help="Size of ensemble of variables"
    )


def _build_argparser():
    parser = argparse.ArgumentParser(description=_ERT3_DESCRIPTION)
    subparsers = parser.add_subparsers(dest="sub_cmd", help="ert3 commands")

    subparsers.add_parser("init", help="Initialize an ERT3 workspace")
    _build_record_argparser(subparsers)

    return parser


def _record(workspace, args):
    assert args.sub_cmd == "record"
    if args.sub_record_cmd == "sample":
        ert3.engine.sample_record(
            workspace, args.parameter_group, args.record_name, args.ensemble_size
        )
    elif args.sub_record_cmd == "load":
        ert3.engine.load_record(workspace, args.record_name, args.record_file)
    else:
        raise NotImplementedError(
            f"No implementation to handle record command {args.sub_record_cmd}"
        )


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

    if args.sub_cmd == "record":
        _record(workspace, args)
    else:
        raise NotImplementedError(f"No implementation to handle command {args.sub_cmd}")
