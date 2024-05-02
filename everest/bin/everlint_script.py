#!/usr/bin/env python

import argparse
from functools import partial

from everest.config import EverestConfig


def lint_entry(args=None):
    arg_parser = argparse.ArgumentParser(
        description="Check if a config file is valid",
        usage="""everest lint <config_file>""",
    )
    arg_parser.add_argument(
        "config_file",
        type=partial(EverestConfig.load_file_with_argparser, parser=arg_parser),
        help="The path to the everest configuration file",
    )

    options = arg_parser.parse_args(args)
    parsed_config = options.config_file
    conf_file = parsed_config.config_path

    print(f"{conf_file} is valid")


if __name__ == "__main__":
    lint_entry()
