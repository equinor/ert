import argparse
import logging
import sys
from functools import partial
from importlib.resources import files
from pathlib import Path
from signal import SIG_DFL, SIGINT, signal
from textwrap import dedent

from PyQt6.QtCore import QDir
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from ert.gui.tools.plot.plot_window import PlotWindow
from ert.services import ErtServerController
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
    arg_parser.add_argument(
        "--ert_plotter",
        action="store_true",
        help="Use the Ert plotter instead of everviz (EXPERIMENTAL)",
    )
    return arg_parser


def visualization_entry(args: list[str] | None = None) -> None:
    parser = _build_args_parser()
    options = parser.parse_args(args)
    logger = logging.getLogger(__name__)
    with setup_logging(options):
        logger.info(
            f"Starting everest results entrypoint with args {args} in {Path.cwd()}"
        )
        ever_config = options.config
        EverestStorage.check_for_deprecated_seba_storage(
            ever_config.optimization_output_dir
        )

        if LocalStorage.check_migration_needed(Path(ever_config.storage_dir)):
            logger.info("Migrating ERT storage from everviz entrypoint")
            LocalStorage.perform_migration(Path(ever_config.storage_dir))

        experiment = EverestStorage.get_everest_experiment(
            storage_path=ever_config.storage_dir,
        )

        if not experiment.ensembles_with_function_results:
            print(
                f"No data found in storage at {experiment._storage.path}. "
                f"Please try again later"
            )
            return

        if not options.ert_plotter:
            pm = EverestPluginManager()
            pm.hook.visualize_data(api=EverestDataAPI(options.config))
        else:
            with ErtServerController.init_service(
                timeout=240,
                project=Path(ever_config.storage_dir),
            ):
                run_plotter_gui(options.config.config_path, ever_config.storage_dir)


def run_plotter_gui(config_filename: str, storage_dir: Path) -> None:
    # Replace Python's exception handler for SIGINT with the system default.
    #
    # Python's SIGINT handler is the one that raises KeyboardInterrupt. This is
    # okay normally (if a bit ugly), but when control is given to Qt this
    # exception handler will either get deadlocked because Python never gets
    # control back, or gets eaten by Qt because it ignores exceptions that
    # happen in Qt slots.
    signal(SIGINT, SIG_DFL)

    QDir.addSearchPath(
        "img", str(files("ert.gui").joinpath("../../ert/gui/resources/gui/img"))
    )

    app = QApplication(["everest"])
    app.setWindowIcon(QIcon("img:ert_icon.svg"))
    window = PlotWindow(f"{config_filename}", storage_dir, parent=None)
    window.adjustSize()
    window.show()
    window.activateWindow()
    window.raise_()
    app.exec()


if __name__ == "__main__":
    visualization_entry(sys.argv[1:])
