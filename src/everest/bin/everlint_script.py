#!/usr/bin/env python

import argparse
import warnings
from functools import partial

from ert.config import ConfigWarning
from everest.config import EverestConfig


def _build_args_parser() -> argparse.ArgumentParser:
    """Build arg parser"""
    arg_parser = argparse.ArgumentParser(
        description="Check if a config file is valid",
        usage="""everest lint <config_file>""",
    )
    arg_parser.add_argument(
        "config",
        type=partial(EverestConfig.load_file_with_argparser, parser=arg_parser),
        help="The path to the everest configuration file",
    )
    arg_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Display verbose errors and warnings",
    )
    return arg_parser


def lint_entry(args: list[str]) -> None:
    if not any("-v" in arg or "--verbose" in arg for arg in args):
        warnings.filterwarnings("ignore")
        warnings.filterwarnings("default", category=ConfigWarning)
    parser = _build_args_parser()
    options = parser.parse_args(args)
    parsed_config = options.config
    conf_file = parsed_config.config_path

    print(f"{conf_file} is valid")


if __name__ == "__main__":
    lint_entry([])
