#!/usr/bin/env python

import argparse
import logging
from functools import partial

from everest.bin.utils import setup_logging
from everest.config import EverestConfig
from everest.strings import EVEREST

logger = logging.getLogger(EVEREST)


def everexport_entry(args: list[str] | None = None) -> None:
    parser = _build_args_parser()
    options = parser.parse_args(args)
    with setup_logging(options):
        logger.info("Everexport deprecation warning seen")
        print(
            "Everexport is deprecated, optimization results "
            f"already exist @ {options.config.optimization_output_dir}"
        )


def _build_args_parser() -> argparse.ArgumentParser:
    """Build arg parser"""
    arg_parser = argparse.ArgumentParser(
        description="Export data from a completed optimization case",
        usage="everest export <config_file>",
    )
    arg_parser.add_argument(
        "config",
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
