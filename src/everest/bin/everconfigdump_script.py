#!/usr/bin/env python

import argparse
import sys

from ruamel.yaml import YAML

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
        "config_file", help="The path to the everest configuration file"
    )
    return arg_parser


def config_dump_entry(args: list[str] | None = None) -> None:
    parser = _build_args_parser()
    options = parser.parse_args(args)

    config = EverestConfig.load_file(options.config_file)

    yaml = YAML(typ="safe", pure=True)
    yaml.indent = 2
    yaml.default_flow_style = False
    yaml.dump(config.to_dict(), sys.stdout)


if __name__ == "__main__":
    config_dump_entry()
