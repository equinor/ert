#!/usr/bin/env python

import argparse
import json
import sys


def main(argv):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--controls-file", type=str, required=True)
    arg_parser.add_argument("--out-prefix", type=str, default="")
    arg_parser.add_argument("--out-suffix", type=str, default="")
    opts, _ = arg_parser.parse_known_args(args=argv)

    with open(opts.controls_file, "r", encoding="utf-8") as f:
        controls = json.load(f)

    for k, v in controls.items():
        out_file = "{}{}{}".format(opts.out_prefix, k, opts.out_suffix)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("%g \n" % v)


if __name__ == "__main__":
    main(sys.argv[1:])
