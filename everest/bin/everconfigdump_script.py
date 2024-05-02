#!/usr/bin/env python

import argparse
import sys

from ruamel.yaml import YAML

from everest.config import EverestConfig


def config_dump_entry(args=None):
    arg_parser = argparse.ArgumentParser(
        description="Print to console the contents of the config file",
        usage="""everest show <config_file>""",
    )
    arg_parser.add_argument(
        "config_file", help="The path to the everest configuration file"
    )
    options = arg_parser.parse_args(args)

    config = EverestConfig.load_file(options.config_file)

    yaml = YAML(typ="safe", pure=True)
    yaml.indent = 2
    yaml.default_flow_style = False
    yaml.dump(config.to_dict(), sys.stdout)


if __name__ == "__main__":
    config_dump_entry()
