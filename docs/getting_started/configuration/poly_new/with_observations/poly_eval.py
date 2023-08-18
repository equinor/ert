#!/usr/bin/env python

import json

with open("coeffs.json", encoding="utf-8") as f:
    coeffs = json.load(f)


def evaluate(coeffs, x):
    return coeffs["a"] * x**2 + coeffs["b"] * x + coeffs["c"]


output = [evaluate(coeffs, x) for x in range(10)]
with open("poly.out", "w", encoding="utf-8") as f:
    f.write("\n".join(map(str, output)))
