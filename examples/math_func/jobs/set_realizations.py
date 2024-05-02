#!/usr/bin/env python

import argparse
import json
import sys

import numpy

# This script is used in test_ensemble_evaluation.py, which uses the
# experimental plan feature. The evaluation step in that plan evaluates the full
# ensemble with the current controls and calls this workflow job with the
# results. This script checks the objectives, calculates new realization
# weights, and writes those to a file. The evaluation job reads those back and
# adjusts the optimizer configuration accordingly.


def main(argv):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-i", "--input", type=str, required=True)
    arg_parser.add_argument("-o", "--output", type=str)
    options, _ = arg_parser.parse_known_args(args=argv)

    with open(options.input, "r", encoding="utf-8") as f:
        results = json.load(f)

    # The test does a restart, and the index should be recorded:
    objectives = numpy.array(results["objectives"]["distance"])
    weights = numpy.array(results["realization_weights"])

    if options.output:
        # Disable realizations with negative objectives:
        new_weights = numpy.where(objectives > 0, 0.0, weights)

        with open(options.output, "w", encoding="utf-8") as f:
            json.dump({"realization_weights": new_weights.tolist()}, f)


if __name__ == "__main__":
    main(sys.argv[1:])
