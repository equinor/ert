#!/usr/bin/env python
import argparse
import ecl2df
import json
import sys
import pathlib


def _build_arg_parser():
    arg_parser = argparse.ArgumentParser(
        description="Exports summary data as JSON",
    )
    arg_parser.add_argument(
        "--datafile",
        type=pathlib.Path,
        required=True,
        help="Path to datafile"
    )
    arg_parser.add_argument(
        "--output",
        type=argparse.FileType('w'),
        required=True,
        help="Path to the output file"
    )
    return arg_parser


def _load_summary(datafile):
    if not datafile.is_file():
        sys.exit(f"{datafile} is not an existing file")
    eclfiles = ecl2df.EclFiles(datafile)
    ecl2df.summary.df(eclfiles)
    return ecl2df.summary.df(eclfiles)


def _summary2json(sdf):
    s = {}
    s["DATE"] = [date.isoformat() for date in sdf.index]
    for key in sdf.columns:
        s[key] = list(map(float, sdf[key]))
    return s


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    summary_data = _load_summary(args.datafile)
    json_summary = _summary2json(summary_data)
    json.dump(json_summary, args.output)
