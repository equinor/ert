#!/usr/bin/env python

import argparse
import json
import sys
from pathlib import Path


def main(argv):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--controls-file", type=str, required=True)
    arg_parser.add_argument("--out-prefix", type=str, default="")
    arg_parser.add_argument("--out-suffix", type=str, default="")
    opts, _ = arg_parser.parse_known_args(args=argv)

    with open(opts.controls_file, encoding="utf-8") as f:
        controls = json.load(f)

    for name, indices in controls.items():
        for idx, val in indices.items():
            out_file = "{}{}{}".format(
                opts.out_prefix, f"{name}-{idx}", opts.out_suffix
            )
            Path(out_file).write_text(f"{val:g} \n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
