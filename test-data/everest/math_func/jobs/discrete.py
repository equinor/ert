#!/usr/bin/env python

import argparse
import json
import sys
from pathlib import Path


def compute_func(x, y):
    return min(3 * x, y)


def read_point(filename):
    with open(filename, encoding="utf-8") as f:
        point = json.load(f)
    return point["x"], point["y"]


def main(argv):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--point-file", type=str)
    arg_parser.add_argument("--out", type=str)
    options, _ = arg_parser.parse_known_args(args=argv)

    point = read_point(options.point_file)
    if len(point) != 2:
        raise RuntimeError("Failed parsing point")

    value = compute_func(*point)

    if options.out:
        Path(options.out).write_text(f"{value:g} \n", encoding="utf-8")

    else:
        print(value)


if __name__ == "__main__":
    main(sys.argv[1:])
