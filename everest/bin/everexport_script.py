#!/usr/bin/env python

import argparse
import logging
from functools import partial

from everest import export_to_csv, validate_export
from everest.config import EverestConfig
from everest.config.export_config import ExportConfig
from everest.strings import EVEREST


def everexport_entry(args=None):
    parser = _build_args_parser()
    options = parser.parse_args(args)
    logger = logging.getLogger(EVEREST)
    if options.debug:
        logger.setLevel(logging.DEBUG)
        # Remove the null handler if set:
        logging.getLogger().removeHandler(logging.NullHandler())

    config = options.config_file

    # Turn into .export once
    # explicit None is disallowed
    if config.export is None:
        config.export = ExportConfig()

    if options.batches is not None:
        batch_list = [int(item) for item in options.batches]
        config.export.batches = batch_list

    err_msgs, export_ecl = validate_export(config)
    for msg in err_msgs:
        logger.warning(msg)
    export_to_csv(config, export_ecl=export_ecl)


def _build_args_parser():
    """Build arg parser"""
    arg_parser = argparse.ArgumentParser(
        description="Export data from a completed optimization case",
        usage="everest export <config_file>",
    )
    arg_parser.add_argument(
        "config_file",
        type=partial(EverestConfig.load_file_with_argparser, parser=arg_parser),
        help="The path to the everest configuration file",
    )
    arg_parser.add_argument(
        "-b",
        "--batches",
        nargs="*",
        help="List of batches to be exported",
    )
    arg_parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Display debug information in the terminal",
    )

    return arg_parser


if __name__ == "__main__":
    everexport_entry()
