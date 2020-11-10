#!/usr/bin/env python
import argparse


def parse_args(args=None):
    arg_parser = argparse.ArgumentParser(
        description="Unix step test script",
    )
    arg_parser.add_argument("argument", help="Expected argument by test script")
    return arg_parser.parse_args(args)


def write_to_file(file, text):
    with open(file, "w") as f:
        f.write(text)


if __name__ == "__main__":
    options = parse_args()
    text_msg = f"Executed unix test script with argument {options.argument}"
    write_to_file("output.out", text_msg)
