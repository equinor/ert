#!/usr/bin/env python
import sys

sum_mock = {}
sum_mock["FPR"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
sum_mock["TIME"] = [
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    110,
    120,
    130,
    140,
    150,
    160,
]


def compute_fpr_diff(summary):
    fpr = summary.get("FPR")
    return fpr[-1] - fpr[0]


def main(argv):
    _ = argv[0]  # input file
    target_file = argv[1]

    diff = compute_fpr_diff(sum_mock)
    with open(target_file, "w", encoding="utf-8") as out:
        out.write("{} \n".format(diff))


if __name__ == "__main__":
    main(sys.argv[1:])
