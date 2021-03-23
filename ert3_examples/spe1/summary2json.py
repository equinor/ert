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
        "--datafile", type=pathlib.Path, required=True, help="Path to datafile"
    )
    arg_parser.add_argument(
        "--keywords", type=str, nargs="*", required=True, help="Keywords to export"
    )
    return arg_parser


def _load_summary(datafile):
    if not datafile.is_file():
        sys.exit(f"{datafile} is not an existing file")
    eclfiles = ecl2df.EclFiles(datafile)
    ecl2df.summary.df(eclfiles)
    return ecl2df.summary.df(eclfiles)


def _summary2json(sdf, keywords):
    s = {}
    index = [date.isoformat() for date in sdf.index]
    for key in keywords:
        values = list(map(float, sdf[key]))
        assert len(index) == len(values)
        s[key] = {idx: val for idx, val in zip(index, values)}
    return s


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    summary_data = _load_summary(args.datafile)
    json_summary = _summary2json(summary_data, args.keywords)

    for keyword, vector in json_summary.items():
        with open(f"{keyword}.json", "w") as f:
            json.dump(vector, f)
