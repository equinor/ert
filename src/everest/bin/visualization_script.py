#!/usr/bin/env python

import argparse
from functools import partial

from everest.api import EverestDataAPI
from everest.config import EverestConfig, ServerConfig
from everest.detached import ServerStatus, everserver_status
from everest.everest_storage import EverestStorage
from everest.plugins.everest_plugin_manager import EverestPluginManager


def _build_args_parser() -> argparse.ArgumentParser:
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


def visualization_entry(args: list[str] | None = None) -> None:
    parser = _build_args_parser()
    options = parser.parse_args(args)
    config = options.config_file

    EverestStorage.check_for_deprecated_seba_storage(config.config_path)
    server_state = everserver_status(
        ServerConfig.get_everserver_status_path(config.output_dir)
    )
    if server_state["status"] != ServerStatus.never_run:
        pm = EverestPluginManager()
        pm.hook.visualize_data(api=EverestDataAPI(config))


if __name__ == "__main__":
    visualization_entry()
