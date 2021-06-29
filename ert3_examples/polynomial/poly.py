#!/usr/bin/env python3
import argparse
import json
import pathlib


def _build_arg_parser():
    def valid_coefficients(file_path):
        path = pathlib.Path(file_path)
        if path.is_file():
            with open(path) as f:
                coefficients = json.load(f)
            return coefficients
        raise argparse.ArgumentTypeError(f"No such file {str(path)}")

    def valid_x_uncertainties(file_path):
        path = pathlib.Path(file_path)
        if path.is_file():
            with open(path) as f:
                x_uncertainties = json.load(f)
            return x_uncertainties
        raise argparse.ArgumentTypeError(f"No such file {str(path)}")

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
        "--x_uncertainties",
        type=valid_x_uncertainties,
        required=False,
        help="Path to file containing the uncertainties for x",
    )
    arg_parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Path to the output file",
    )
    return arg_parser


def _evaluate_polynomial(coefficients, x_uncertainties):
    a, b, c = coefficients["a"], coefficients["b"], coefficients["c"]
    if x_uncertainties:
        x_range = range(len(x_uncertainties))
        xs = map(sum, zip(tuple(x_range), x_uncertainties))
    else:
        x_range = range(10)
        xs = tuple(x_range)
    return tuple(a * x ** 2 + b * x + c for x in xs)


def _main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    result = _evaluate_polynomial(args.coefficients, args.x_uncertainties)
    with open(args.output, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    _main()
