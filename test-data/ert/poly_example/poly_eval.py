#!/usr/bin/env python3
import json
from pathlib import Path


def _evaluate(coeffs, x):
    return coeffs["a"]["value"] * x**2 + coeffs["b"]["value"] * x + coeffs["c"]["value"]


if __name__ == "__main__":
    coeffs = json.loads(Path("parameters.json").read_text(encoding="utf-8"))
    output = [_evaluate(coeffs, x) for x in range(10)]
    Path("poly.out").write_text("\n".join(map(str, output)), encoding="utf-8")
