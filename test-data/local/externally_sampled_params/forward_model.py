#!/usr/bin/env python
import os
import json

import numpy as np

rng = np.random.default_rng()


def _load_coeffs(filename):
    with open(filename) as f:
        return json.load(f)


def _evaluate(coeffs, x):
    return coeffs["a"] * x**2 + coeffs["b"] * x + coeffs["c"]


if __name__ == "__main__":
    a = rng.uniform(0, 1)
    b = rng.uniform(0, 2)
    c = rng.uniform(0, 5)

    with open("externally_sampled_params", "w") as f:
        f.write(f"{a}\n{b}\n{c}")

    # Load updated from parameters if they exist,
    # otherwise use the parameters sampled above.
    if os.path.exists("params.json"):
        coeffs = _load_coeffs("params.json")
    else:
        coeffs = {"a": a, "b": b, "c": c}

    output = [_evaluate(coeffs, x) for x in range(10)]
    with open("external_samples_0.out", "w") as f:
        f.write("\n".join(map(str, output)))
