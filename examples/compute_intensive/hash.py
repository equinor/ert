#!/usr/bin/env python
import argparse
import json
import hashlib

from itertools import product
from string import ascii_lowercase, ascii_uppercase, digits
from operator import add
from functools import reduce
from time import perf_counter


def _build_arg_parser():
    arg_parser = argparse.ArgumentParser(
        description="Try to brute force hashes",
    )
    arg_parser.add_argument(
        "--input",
        type=argparse.FileType("r"),
        required=True,
        help="Path to the input file",
    )
    arg_parser.add_argument(
        "--output",
        type=argparse.FileType("w"),
        required=True,
        help="Path to the output file",
    )
    return arg_parser


# creates hashes until timeout
def _compute_hashes(timeout_seconds: int = 10) -> dict:
    start_time = perf_counter()  # seconds
    max_try_length = 100
    for l in range(1, int(max_try_length) + 1):
        for e in product(ascii_lowercase + ascii_uppercase + digits, repeat=l):
            plaintext = reduce(add, e)
            hashlib.sha256(plaintext.encode("utf-8")).hexdigest()

            if perf_counter() - start_time > timeout_seconds:
                return dict({"time": timeout_seconds})


def _main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    input = json.load(args.input)

    timeout = int(input["timeout_seconds"])

    result = _compute_hashes(timeout)
    json.dump(result, args.output)


if __name__ == "__main__":
    _main()
