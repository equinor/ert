#!/usr/bin/env python

import argparse
import sys


def main(argv):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-o", "--out", type=str, required=True)
    arg_parser.add_argument("-m", "--message", type=str)
    options, _ = arg_parser.parse_known_args(args=argv)

    msg = options.message if options.message else "test"
    with open(options.out, "w", encoding="utf-8") as f:
        f.write(f"{msg}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
