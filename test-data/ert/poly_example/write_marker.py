#!/usr/bin/env python3
import argparse
import pathlib
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("text")
    parser.add_argument("sleep", type=float)
    args = parser.parse_args()
    time.sleep(args.sleep)
    pathlib.Path(args.output).write_text(f"{args.text}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
