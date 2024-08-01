#!/usr/bin/env python

import argparse
from functools import partial

from everest.api import EverestDataAPI
from everest.config import EverestConfig
from everest.detached import ServerStatus, everserver_status
from everest.plugins.hook_manager import EverestPluginManager


def _build_args_parser():
    arg_parser = argparse.ArgumentParser(
        description="Start possible plugin containing everest visualization",
        usage="""everest results <config_file>""",
    )
    arg_parser.add_argument(
        "config_file",
        type=partial(EverestConfig.load_file_with_argparser, parser=arg_parser),
        help="The path to the everest configuration file",
    )
    return arg_parser


def visualization_entry(args=None):
    parser = _build_args_parser()
    options = parser.parse_args(args)
    config = options.config_file

    server_state = everserver_status(config)
    if server_state["status"] != ServerStatus.never_run:
        pm = EverestPluginManager()
        pm.hook.visualize_data(api=EverestDataAPI(config))


if __name__ == "__main__":
    visualization_entry()
