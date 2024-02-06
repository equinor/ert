#!/usr/bin/env python3
import json
import random


def _load_coeffs(filename):
    with open(filename, encoding="utf-8") as f:
        return json.load(f)["COEFFS"]


def _evaluate(coeffs, x, noise):
    return (
        (coeffs["a"] + noise) * x**2 + (coeffs["b"] + noise) * x + coeffs["c"] + noise
    )


if __name__ == "__main__":
    for rs in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        noise = random.randint(1, 10) / 10
        coeffs = _load_coeffs("parameters.json")
        output = [_evaluate(coeffs, x, noise) for x in range(10)]
        with open(f"poly{rs}.out", "w", encoding="utf-8") as f:
            f.write("\n".join(map(str, output)))
