#!/usr/bin/env python
import argparse
import json
import os

from time import perf_counter


def _build_arg_parser():
    arg_parser = argparse.ArgumentParser(
        description="Read and write files locally",
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


def _io_raw(num_files: int, file_size: int) -> dict:
    start_time = perf_counter()  # seconds
    bytes_read = 0
    bytes_written = 0

    for e in range(0, num_files):
        file_name = f"{e}.out"
        with open(file_name, "wb") as f:
            # if file_size == 0:
            # random
            f.write(os.urandom(file_size))
            bytes_written += file_size

        with open(file_name, "rb") as f:
            s = f.read()
            bytes_read += len(s)

    t = perf_counter() - start_time

    return dict(
        {
            "files_written": num_files,
            "files_read": num_files,
            "total_written_bytes": bytes_written,
            "total_read_bytes": bytes_read,
            "total_time": t,
        }
    )


def _main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    input = json.load(args.input)

    num_files = int(input["num_files"])
    file_size = int(input["file_size"])

    result = _io_raw(num_files, file_size)
    json.dump(result, args.output)


if __name__ == "__main__":
    _main()
