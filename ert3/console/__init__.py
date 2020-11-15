import argparse
import os
import pathlib
import sys


def _locate_ert_workspace_root(path):
    path = pathlib.Path(path)
    while True:
        if os.path.exists(path/".ert"):
            return path
        if path == pathlib.Path(path.root):
            return None
        path = path.parent


_ERT3_DESCRIPTION = (
    "ert3 is an ensemble-based tool for uncertainty studies.\n"
    "\nWARNING: the tool is currently in an extremely early stage and we refer "
    "all users to ert for real work!"
)


def _build_argparser():
    parser = argparse.ArgumentParser(description=_ERT3_DESCRIPTION)
    subparsers = parser.add_subparsers(dest="sub_cmd", help="ert3 commands")

    init_parser = subparsers.add_parser("init", help="Initialize an ERT3 workspace")

    return parser


def _init_workspace(path):
    path = pathlib.Path(path)
    if _locate_ert_workspace_root(path) is not None:
        sys.exit("Already inside an ERT workspace")

    with open(path/".ert", "w") as fout:
        fout.write("ERT workspace")


def main():
    parser = _build_argparser()
    args = parser.parse_args()

    if args.sub_cmd is None:
        parser.print_help()
    elif args.sub_cmd == "init":
        _init_workspace(os.getcwd())
    else:
        sys.exit("Not inside an ERT workspace")
