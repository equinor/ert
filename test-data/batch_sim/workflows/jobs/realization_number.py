#!/usr/bin/env python
import argparse
import re
from pathlib import Path

TARGET_FILE = "realization.number"
REGEX = r"realization-(\d+)"


def add_file_to_realization_runpaths(runpath_file):
    with open(runpath_file, "r", encoding="utf-8") as fh:
        runpath_file_lines = fh.readlines()

    for line in runpath_file_lines:
        realization_path = line.split()[1]
        realization_nr = re.findall(REGEX, realization_path)
        (Path(realization_path) / TARGET_FILE).write_text(
            f"{realization_nr[0]}", encoding="utf-8"
        )


def job_parser():
    description = (
        "A workflow job that, if the komodo version is set, "
        "adds a file with the current Komodo version to the "
        "runpath of each realization."
    )
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
