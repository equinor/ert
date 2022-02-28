#!/usr/bin/env python

coeffs = {"a": 1, "b": 2, "c": 3}


def evaluate(coeffs, x):
    return coeffs["a"] * x**2 + coeffs["b"] * x + coeffs["c"]


output = [evaluate(coeffs, x) for x in range(10)]
with open("poly_0.out", "w") as f:
    f.write("\n".join(map(str, output)))
