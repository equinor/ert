#!/usr/bin/env python
import json
import sys


def _load_coeffs(filename):
    with open(filename, encoding="utf-8") as f:
        return json.load(f)


def _evaluate(coeffs, x):
    return coeffs["a"] * x**2 + coeffs["b"] * x + coeffs["c"]


if __name__ == "__main__":
    args = sys.argv[1:]
    print(f"Running {sys.argv[0]}")
    if len(args) == 0:
        sys.exit("My error message!!!!")
    coeffs = _load_coeffs("coeffs.json")
    output = [_evaluate(coeffs, x) for x in range(10)]
    with open("poly_0.out", "w", encoding="utf-8") as f:
        f.write("\n".join(map(str, output)))
