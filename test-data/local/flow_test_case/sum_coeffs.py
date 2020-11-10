#!/usr/bin/env python
import numpy as np


def _load_coeffs(filename):
    with open(filename) as f:
        return np.array([float(l.strip("\n")) for l in f.readlines()])


def write_to_file(data, file):
    with open(file, 'w') as f:
        f.write('\n'.join(map(str, data)))


if __name__ == '__main__':
    coeffs_0 = _load_coeffs('poly_0.out')
    coeffs_1 = _load_coeffs('poly_1.out')
    coeffs_2 = _load_coeffs('poly_2.out')
    coeffs_sum = coeffs_0 + coeffs_1 + coeffs_2

    write_to_file(coeffs_sum, f"poly_sum.out")
