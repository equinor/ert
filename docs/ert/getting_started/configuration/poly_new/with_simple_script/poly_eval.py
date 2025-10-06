#!/usr/bin/env python
from pathlib import Path

coeffs = {"a": 1, "b": 2, "c": 3}


def evaluate(coeffs, x):
    return coeffs["a"] * x**2 + coeffs["b"] * x + coeffs["c"]


output = [evaluate(coeffs, x) for x in range(10)]
Path("poly.out").write_text("\n".join(map(str, output)), encoding="utf-8")
