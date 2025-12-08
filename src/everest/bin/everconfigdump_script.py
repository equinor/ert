#!/usr/bin/env python

import argparse
import sys

from everest.config import EverestConfig


def _build_args_parser() -> argparse.ArgumentParser:
    """Build arg parser"""
    arg_parser = argparse.ArgumentParser(
        description=(
            "Display the configuration data loaded from a "
            "config file after replacing templated arguments."
        ),
        usage="""everest show <config_file>""",
    )
    arg_parser.add_argument(
        "config_file", help="The path to the everest configuration file."
    )
    return arg_parser


def config_dump_entry(args: list[str] | None = None) -> None:
    parser = _build_args_parser()
    options = parser.parse_args(args)
    EverestConfig.load_file(options.config_file).write_to_file(sys.stdout)


if __name__ == "__main__":
    config_dump_entry()
