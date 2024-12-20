import argparse

#!/usr/bin/env python3
import json


def _create_config(filename, a, b, c):
    with open(filename, encoding="utf-8", mode="w+") as f:
        json.dump({"checksum": a * b * c}, f)


def main_entry_point():
    parser = argparse.ArgumentParser(description="Creates a poly config")

    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to the config file"
    )

    parser.add_argument("a", type=float, help="a", default=1, nargs="?")

    parser.add_argument("b", type=float, help="b", default=2, nargs="?")

    parser.add_argument("c", type=float, help="c", default=3, nargs="?")
    try:
        args = parser.parse_args()
        _create_config(args["-c"], args.a, args.b, args.c)
    except Exception as err:
        raise SystemExit(str(err)) from err


if __name__ == "__main__":
    main_entry_point()
