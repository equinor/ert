#!/usr/bin/env python

import json

import numpy as np


def _load_coeffs(filename):
    with open(filename) as f:
        return json.load(f)


def _evaluate(coeffs):
    K = 2500
    x = np.zeros(K)
    x[0] = 0
    x[1] = 1

    # Looping from 2 because we have initial conditions at k=0 and k=1.
    for k in range(2, K - 1):
        M = np.array(
            [[2 + coeffs["omega"] ** 2 - coeffs["lmbda"] ** 2 * x[k] ** 2, -1], [1, 0]]
        )
        u = np.array([x[k], x[k - 1]])
        u = M @ u
        x[k + 1] = u[0]
        x[k] = u[1]

    return x


if __name__ == "__main__":
    coeffs = _load_coeffs("coeffs.json")
    output = _evaluate(coeffs)
    np.savetxt("oscillator_0.out", output, fmt="%f")
