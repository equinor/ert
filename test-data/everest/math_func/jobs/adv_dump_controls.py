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

    with open(opts.controls_file, encoding="utf-8") as f:
        controls = json.load(f)

    for name, indices in controls.items():
        for idx, val in indices.items():
            out_file = "{}{}{}".format(
                opts.out_prefix, "{}-{}".format(name, idx), opts.out_suffix
            )
            with open(out_file, "w", encoding="utf-8") as f:
                f.write("%g \n" % val)


if __name__ == "__main__":
    main(sys.argv[1:])
