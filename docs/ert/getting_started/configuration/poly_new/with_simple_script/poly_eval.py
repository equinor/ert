#!/usr/bin/env python
from pathlib import Path

coeffs = {"a": 1.0, "b": 2.0, "c": 3.0}


def evaluate(coeffs: dict[str, float], x: float) -> float:
    return coeffs["a"] * x**2 + coeffs["b"] * x + coeffs["c"]


output = [evaluate(coeffs, x) for x in range(10)]
Path("poly.out").write_text("\n".join(map(str, output)), encoding="utf-8")
