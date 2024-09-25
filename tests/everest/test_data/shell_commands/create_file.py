#!/usr/bin/env python

import argparse


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--name", "-n", type=str, required=True)
    options = arg_parser.parse_args()

    with open(options.name, "w", encoding="utf-8") as f:
        f.write("Let there be a file!")


if __name__ == "__main__":
    main()
