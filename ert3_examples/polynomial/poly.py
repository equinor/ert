#!/usr/bin/env python
import argparse
import json
import sys


def _build_arg_parser():
    arg_parser = argparse.ArgumentParser(
        description="Computes the result of a second degree polynomial",
    )
    arg_parser.add_argument(
        "--coefficients",
        type=argparse.FileType("r"),
        required=True,
        help="Path to file containing the coefficients",
    )
    arg_parser.add_argument(
        "--output",
        type=argparse.FileType("w"),
        required=True,
        help="Path to the output file",
    )
    return arg_parser


def _evaluate_polynomial(coefficients):
    a, b, c = coefficients["a"], coefficients["b"], coefficients["c"]
    x_range = tuple(range(10))
    return tuple(a * x ** 2 + b * x + c for x in x_range)


def _main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    coefficients = json.load(args.coefficients)
    result = _evaluate_polynomial(coefficients)
    json.dump(result, args.output)


if __name__ == "__main__":
    _main()
