#!/usr/bin/env python
import argparse
import json
import time


def parse_args(args=None):
    arg_parser = argparse.ArgumentParser(
        description="Unix step test script",
    )
    arg_parser.add_argument("argument", help="Expected argument by test script")
    return arg_parser.parse_args(args)


def write_to_file(file, data):
    with open(file, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    options = parse_args()

    # this sleep prevents regressions like https://github.com/equinor/ert/issues/2756
    # in conjunction with aggressive ping_interval and ping_timout in the evaluation
    # server
    time.sleep(2)

    data = [1, 2, 3]
    write_to_file("output.out", data)
