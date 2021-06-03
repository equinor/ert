#!/usr/bin/env python3
import argparse
import json
import sys
import pathlib


def _build_arg_parser():
    def valid_coefficients(file_path):
        path = pathlib.Path(file_path)
        if path.exists():
            with open(path, "r") as f:
                coefficients = json.load(f)
            return coefficients
        raise argparse.ArgumentTypeError(f"No such file or directory {str(path)}")

    arg_parser = argparse.ArgumentParser(
        description="Computes the result of a second degree polynomial",
    )
    arg_parser.add_argument(
        "--coefficients",
        type=valid_coefficients,
        required=True,
        help="Path to file containing the coefficients",
    )
    arg_parser.add_argument(
        "--output",
        type=pathlib.Path,
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
    result = _evaluate_polynomial(args.coefficients)
    with open(args.output, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    _main()
