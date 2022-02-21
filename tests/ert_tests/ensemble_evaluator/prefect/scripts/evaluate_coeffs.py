#!/usr/bin/env python
import json
import argparse


def _load_coeffs(filename):
    with open(filename) as f:
        return json.load(f)


def _evaluate(coeffs, x, degree):
    if degree == 0:
        return coeffs["c"]
    if degree == 1:
        return coeffs["b"] * x
    if degree >= 2:
        return coeffs["a"] * x**degree


def config_dump_entry(args=None):
    arg_parser = argparse.ArgumentParser(
        description="Degree to compute",
    )
    arg_parser.add_argument(
        "degree", type=int, help="What polynomial component to compute"
    )
    return arg_parser.parse_args(args)


def write_to_file(data, file):
    with open(file, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    options = config_dump_entry()
    coeffs = _load_coeffs("coeffs.json")
    print(f"Calculating {options.degree} degree polynomial component")
    output = [_evaluate(coeffs, x, options.degree) for x in range(10)]
    print(f"Writing output to file poly_{options.degree}.out")
    write_to_file(output, f"poly_{options.degree}.out")
