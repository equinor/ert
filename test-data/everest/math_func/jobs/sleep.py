#!/usr/bin/env python

import argparse
import sys
import time


def main(argv):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--sleep", type=int)
    options, _ = arg_parser.parse_known_args(args=argv)
    time.sleep(options.sleep)


if __name__ == "__main__":
    main(sys.argv[1:])
