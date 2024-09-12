#!/usr/bin/env python
import sys


def save_return_value(return_value, target_file):
    """
    Save to file
    :param return_value:
    :param target_file:
    :return:
    """
    with open(target_file, "w", encoding="utf-8") as f:
        f.write("%g \n" % (1.0 * return_value))


def main(argv):
    # Main script starts here
    times = sys.argv[2:]

    sum = {}
    sum["FOPR"] = 17 * [6000]

    for i in range(0, len(times)):
        val = sum.get("FOPR")[i]
        save_return_value(val, "oil_prod_rate_%03d" % i)

    with open("OIL_PROD_RATE_OK", "w", encoding="utf-8") as f:
        f.write("Everything went fine here!")


if __name__ == "__main__":
    main(sys.argv[1:])
