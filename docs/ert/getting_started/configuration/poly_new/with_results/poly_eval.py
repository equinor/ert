#!/usr/bin/env python
import json
from pathlib import Path

with open("parameters.json", encoding="utf-8") as f:
    coeffs = json.load(f)["COEFFS"]


def evaluate(coeffs, x):
    return coeffs["a"] * x**2 + coeffs["b"] * x + coeffs["c"]


output = [evaluate(coeffs, x) for x in range(10)]
Path("poly.out").write_text("\n".join(map(str, output)), encoding="utf-8")
