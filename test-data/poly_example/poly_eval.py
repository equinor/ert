#!/usr/bin/env python3
import json
import time


def _load_coeffs(filename):
    with open(filename, encoding="utf-8") as f:
        return json.load(f)["COEFFS"]


def _evaluate(coeffs, x):
    return coeffs["a"] * x ** 2 + coeffs["b"] * x + coeffs["c"]


if __name__ == "__main__":
    time.sleep(4)
    coeffs = _load_coeffs("parameters.json")
    output = [_evaluate(coeffs, x) for x in range(10)]
    with open("poly.out", "w", encoding="utf-8") as f:
        f.write("\n".join(map(str, output)))
