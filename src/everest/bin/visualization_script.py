import argparse
import logging
import sys
from functools import partial
from pathlib import Path
from textwrap import dedent

from ert.storage import LocalStorage
from everest.api import EverestDataAPI
from everest.bin.utils import setup_logging
from everest.config import EverestConfig
from everest.everest_storage import EverestStorage
from everest.plugins.everest_plugin_manager import EverestPluginManager

from .utils import ArgParseFormatter


def _build_args_parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(
        description=dedent(
            """
            Start an everest visualization plugin.

            If no visualization plugin is installed the message: ``No
            visualization plugin installed!`` will be displayed in the console.

            The recommended open-source everest visualization plugin is Everviz:

            https://github.com/equinor/everviz

            It can be installed using ``pip install everviz``.
            """
        ),
        formatter_class=ArgParseFormatter,
        usage="""everest results <config_file>""",
    )
    arg_parser.add_argument(
        "config",
        type=partial(EverestConfig.load_file_with_argparser, parser=arg_parser),
        help="The path to the everest configuration file.",
    )
    return arg_parser


def visualization_entry(args: list[str] | None = None) -> None:
    parser = _build_args_parser()
    options = parser.parse_args(args)
    logger = logging.getLogger(__name__)
    with setup_logging(options):
        logger.info(f"Starting everviz entrypoint with args {args} in {Path.cwd()}")
        ever_config = options.config
        EverestStorage.check_for_deprecated_seba_storage(
            ever_config.optimization_output_dir
        )

        if LocalStorage.check_migration_needed(Path(ever_config.storage_dir)):
            logger.info("Migrating ERT storage from everviz entrypoint")
            LocalStorage.perform_migration(Path(ever_config.storage_dir))

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
