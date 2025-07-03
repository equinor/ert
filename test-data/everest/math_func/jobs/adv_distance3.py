#!/usr/bin/env python

import argparse
import json
import sys
from pathlib import Path


def compute_distance_squared(
    p: tuple[float, float, float], q: tuple[float, float, float]
) -> float:
    d = ((i - j) ** 2 for i, j in zip(p, q, strict=True))
    return -sum(d)


def read_point(filename: Path) -> tuple[float, float, float]:
    point = json.loads(filename.read_text(encoding="utf-8"))
    x = point["x"]
    return x["0"], x["1"], x["2"]


def main(argv: list[str]) -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--point-file", type=Path)
    arg_parser.add_argument("--point", nargs=3, type=float)
    arg_parser.add_argument("--target-file", type=Path)
    arg_parser.add_argument("--target", nargs=3, type=float)
    arg_parser.add_argument("--out", type=Path, required=True)
    options, _ = arg_parser.parse_known_args(args=argv)

    point = options.point or read_point(options.point_file)
    if len(point) != 3:
        msg = "Failed parsing point"
        raise RuntimeError(msg)

    target = options.target or read_point(options.target_file)
    if len(target) != 3:
        msg = "Failed parsing target"
        raise RuntimeError(msg)

    value = compute_distance_squared(point, target)

    if options.out:
        options.out.write_text(f"{value:g} \n", encoding="utf-8")
    else:
        print(value)


if __name__ == "__main__":
    main(sys.argv[1:])
