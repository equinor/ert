#!/usr/bin/env python

import sys


def main(argv):
    simulation_id = argv[0]
    failures = argv[1:]

    # Generate the objective function file
    with open("mock_objective", "w", encoding="utf-8") as f:
        f.write("0")

    if simulation_id in failures:
        sys.exit("You asked for failure..")


if __name__ == "__main__":
    main(sys.argv[1:])
