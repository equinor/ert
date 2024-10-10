#!/usr/bin/env python
import argparse
import glob
import subprocess
import sys
from pathlib import Path
from typing import Dict


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OPM-Flow shell wrapper")
    parser.add_argument(
        "-v", "--version", default="stable", type=str, help="Version of Flow to run"
    )
    parser.add_argument(
        "--exe-args",
        default="",
        type=str,
        help="Pass a string of arguments directly to the executable",
    )
    parser.add_argument(
        "--report-versions", action="store_true", help="List available versions"
    )
    parser.add_argument("--np", default=1, type=int, help="Number of processors to use")
    parser.add_argument(
        "--threads", default=1, type=int, help="Number of threads per process."
    )
    parser.add_argument("DATAFILE", help="Filename defining the simulation deck.")

    # Mostly to allow an empty string when called from Ert:
    parser.add_argument("other_positional_arguments", nargs="*")

    return parser


def get_available_versions() -> Dict[str, Path]:
    versions = {}

    for possible_flowexecutable in glob.glob("/project/res/x86_64_RH_8/bin/flow*"):
        versions[Path(possible_flowexecutable).name.replace("flow", "")] = Path(
            possible_flowexecutable
        )

    if "latest" not in versions:
        versions["latest"] = versions["daily"]

    if "stable" not in versions:
        versions["stable"] = versions["daily"]

    if "default" not in versions:
        versions["default"] = versions["daily"]

    return versions


def main():
    args = get_parser().parse_args(sys.argv[1:])

    binaries = get_available_versions()

    if args.report_versions:
        for version, path in binaries.items():
            print(f"   {version}: {path}")
        sys.exit(0)

    if args.version not in binaries:
        print(f"ERROR: Version {args.version} not found", file=sys.stderr)
        sys.exit(1)

    subprocess.run([binaries[args.version], args.DATAFILE], check=False)


if __name__ == "__main__":
    # print(sys.argv, file=sys.stderr)
    main()
