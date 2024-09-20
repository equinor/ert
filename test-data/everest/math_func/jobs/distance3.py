#!/usr/bin/env python

import argparse
import json
import sys


def compute_distance_squared(p, q):
    d = ((i - j) ** 2 for i, j in zip(p, q))
    d = sum(d)
    return -d


def read_point(filename):
    with open(filename, "r", encoding="utf-8") as f:
        point = json.load(f)
    return point["x"], point["y"], point["z"]


def main(argv):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--point-file", type=str)
    arg_parser.add_argument("--point", nargs=3, type=float)
    arg_parser.add_argument("--target-file", type=str)
    arg_parser.add_argument("--target", nargs=3, type=float)
    arg_parser.add_argument("--out", type=str)
    arg_parser.add_argument("--scaling", nargs=4, type=float)
    arg_parser.add_argument("--realization", type=float)
    options, _ = arg_parser.parse_known_args(args=argv)

    point = options.point if options.point else read_point(options.point_file)
    if len(point) != 3:
        raise RuntimeError("Failed parsing point")

    target = options.target if options.target else read_point(options.target_file)
    if len(target) != 3:
        raise RuntimeError("Failed parsing target")

    if options.scaling is not None:
        min_range, max_range, target_min, target_max = options.scaling
        point = [(p - target_min) / (target_max - target_min) for p in point]
        point = [p * (max_range - min_range) + min_range for p in point]

    value = compute_distance_squared(point, target)
    # If any realizations with an index > 0 are passed we make those incorrect
    # by taking the negative value. This used by test_cvar.py.
    if options.realization:
        value = -value

    if options.out:
        with open(options.out, "w", encoding="utf-8") as f:
            f.write("%g \n" % value)
    else:
        print(value)


if __name__ == "__main__":
    main(sys.argv[1:])
