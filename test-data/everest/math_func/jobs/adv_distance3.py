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
    x = point["x"]
    return x["0"], x["1"], x["2"]


def main(argv):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--point-file", type=str)
    arg_parser.add_argument("--point", nargs=3, type=float)
    arg_parser.add_argument("--target-file", type=str)
    arg_parser.add_argument("--target", nargs=3, type=float)
    arg_parser.add_argument("--out", type=str)
    options, _ = arg_parser.parse_known_args(args=argv)

    point = options.point if options.point else read_point(options.point_file)
    if len(point) != 3:
        raise RuntimeError("Failed parsing point")

    target = options.target if options.target else read_point(options.target_file)
    if len(target) != 3:
        raise RuntimeError("Failed parsing target")

    value = compute_distance_squared(point, target)

    if options.out:
        with open(options.out, "w", encoding="utf-8") as f:
            f.write("%g \n" % value)
    else:
        print(value)


if __name__ == "__main__":
    main(sys.argv[1:])
