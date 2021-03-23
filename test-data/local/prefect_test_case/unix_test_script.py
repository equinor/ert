#!/usr/bin/env python
import argparse
import json


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
    data = [1, 2, 3]
    write_to_file("output.out", data)
