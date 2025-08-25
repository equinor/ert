#!/usr/bin/env python

import argparse
import logging
import sys
from functools import partial
from pathlib import Path

from ert.storage import ErtStorageException, open_storage
from everest.api import EverestDataAPI
from everest.bin.utils import setup_logging
from everest.config import EverestConfig
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
    with setup_logging(options):
        ever_config = options.config

        EverestStorage.check_for_deprecated_seba_storage(
            ever_config.optimization_output_dir
        )

        try:
            # If successful, no need to migrate
            open_storage(ever_config.storage_dir, mode="r").close()
        except ErtStorageException as err:
            if "too old" in str(err):
                # Open write storage to do a migration
                logging.getLogger(__name__).info(
                    "Migrating ERT storage from everviz entrypoint"
                )
                open_storage(ever_config.storage_dir, mode="w").close()

        storage = EverestStorage(output_dir=Path(ever_config.optimization_output_dir))
        storage.read_from_output_dir()

        if storage.is_empty:
            print(
                f"No data found in storage at {storage._output_dir}."
                " Please try again later"
            )
            return

        pm = EverestPluginManager()
        pm.hook.visualize_data(api=EverestDataAPI(options.config))


if __name__ == "__main__":
    visualization_entry(sys.argv[1:])
