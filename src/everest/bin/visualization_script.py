#!/usr/bin/env python

import argparse
from functools import partial

from everest.api import EverestDataAPI
from everest.bin.utils import setup_logging
from everest.config import EverestConfig, ServerConfig
from everest.detached import ExperimentState, everserver_status
from everest.everest_storage import EverestStorage
from everest.plugins.everest_plugin_manager import EverestPluginManager


def _build_args_parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(
        description="Start possible plugin containing everest visualization",
        usage="""everest results <config_file>""",
    )
    arg_parser.add_argument(
        "config",
        type=partial(EverestConfig.load_file_with_argparser, parser=arg_parser),
        help="The path to the everest configuration file",
    )
    return arg_parser


def visualization_entry(args: list[str] | None = None) -> None:
    parser = _build_args_parser()
    options = parser.parse_args(args)
    setup_logging(options)

    EverestStorage.check_for_deprecated_seba_storage(
        options.config.optimization_output_dir
    )
    server_state = everserver_status(
        ServerConfig.get_everserver_status_path(options.config.output_dir)
    )
    if server_state["status"] in {
        ExperimentState.failed,
        ExperimentState.stopped,
        ExperimentState.completed,
    }:
        pm = EverestPluginManager()
        pm.hook.visualize_data(api=EverestDataAPI(options.config))
    elif server_state["status"] in {
        ExperimentState.running,
        ExperimentState.pending,
    }:
        print(
            "Everest is running, please wait for it to finish before "
            "running the visualization."
        )
    else:
        print("No Everest results found for the given configuration.")


if __name__ == "__main__":
    visualization_entry()
