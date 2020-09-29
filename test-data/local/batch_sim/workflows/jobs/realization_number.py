#!/usr/bin/env python
import os.path
import argparse
import re
TARGET_FILE = "realization.number"
REGEX = "realisation-(\d+)"


def add_file_to_realization_runpaths(runpath_file):
    with open(runpath_file, "r") as fh:
        runpath_file_lines = fh.readlines()

    for line in runpath_file_lines:
        realization_path = line.split()[1]
        with open(
            os.path.join(realization_path, TARGET_FILE), "w"
        ) as fh:
            realization_nr = re.findall(r"realisation-(\d+)", realization_path)
            fh.write("{}\n".format(realization_nr[0]))


def job_parser():
    description = """A workflow job that, if the komodo version is set, adds a file with the current Komodo version to the runpath of each realization."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "runpath_file",
        type=str,
        help="Path to the file containing the runpath of all the realizations.",
    )
    return parser


if __name__ == "__main__":
    parser = job_parser()
    args = parser.parse_args()
    add_file_to_realization_runpaths(args.runpath_file)
