#!/usr/bin/env python

import argparse
import os
import sys
import time


def main(argv):
    """
    This job should only be used for testing! Was designed to fail when running
    a specific simulation.

    Example:
         fail_simulation.py --fail simulation_2. Will fail when the jobs
           is run for simulation_2
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--fail", type=str)
    options, _ = arg_parser.parse_known_args(args=argv)

    if options.fail in os.getcwd():
        raise Exception("Failing %s by request!" % options.fail)

    time.sleep(1)


if __name__ == "__main__":
    main(sys.argv[1:])
