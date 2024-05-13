import logging
import os
import sys
import warnings
import webbrowser
from signal import SIG_DFL, SIGINT, signal
from typing import Optional, cast

if sys.version_info >= (3, 9):
    from importlib.resources import files
else:
    from importlib_resources import files

from PyQt5.QtGui import QIcon
from qtpy.QtCore import QDir, QLocale, Qt
from qtpy.QtWidgets import QApplication

from ert.config import ConfigValidationError, ConfigWarning, ErtConfig
from ert.enkf_main import EnKFMain
from ert.gui.ertwidgets import SummaryPanel
from ert.gui.main_window import ErtMainWindow
from ert.gui.simulation import SimulationPanel
from ert.gui.tools.event_viewer import (
    EventViewerTool,
    GUILogHandler,
    add_gui_log_handler,
)
from ert.gui.tools.export import ExportTool
from ert.gui.tools.load_results import LoadResultsTool
from ert.gui.tools.manage_experiments import ManageExperimentsTool
from ert.gui.tools.plot import PlotTool
from ert.gui.tools.plugins import PluginHandler, PluginsTool
from ert.gui.tools.run_analysis import RunAnalysisTool
from ert.gui.tools.workflows import WorkflowsTool
from ert.libres_facade import LibresFacade
from ert.namespace import Namespace
from ert.services import StorageService
from ert.shared.plugins.plugin_manager import ErtPluginManager
from ert.storage import Storage, open_storage
from ert.storage.local_storage import local_storage_set_ert_config

from .suggestor import Suggestor


def run_gui(args: Namespace, plugin_manager: Optional[ErtPluginManager] = None):
    # Replace Python's exception handler for SIGINT with the system default.
    #
    # Python's SIGINT handler is the one that raises KeyboardInterrupt. This is
    # okay normally (if a bit ugly), but when control is given to Qt this
    # exception handler will either get deadlocked because Python never gets
    # control back, or gets eaten by Qt because it ignores exceptions that
    # happen in Qt slots.
    signal(SIGINT, SIG_DFL)

    QDir.addSearchPath("img", str(files("ert.gui").joinpath("resources/gui/img")))

    app = QApplication([])  # Early so that QT is initialized before other imports
    app.setWindowIcon(QIcon("img:application/window_icon_cutout"))
    with add_gui_log_handler() as log_handler:
        window, ens_path, ensemble_size, parameter_config = _start_initial_gui_window(
            args, log_handler, plugin_manager
        )

        def show_window():
            window.show()
            window.activateWindow()
            window.raise_()
            return app.exec_()

        # ens_path is None indicates that there was an error in the setup and
        # window is now just showing that error message, in which
        # case display it and don't show an error message
        if ens_path is None:
            return show_window()

        with StorageService.init_service(project=os.path.abspath(ens_path)):
            return show_window()


def _start_initial_gui_window(
    args, log_handler, plugin_manager: Optional[ErtPluginManager] = None
):
    # Create logger inside function to make sure all handlers have been added to
    # the root-logger.
    logger = logging.getLogger(__name__)
    error_messages = []
    all_warnings = []
    config_warnings = []
    ert_config = None

    with warnings.catch_warnings(record=True) as all_warnings:
        try:
            _check_locale()
            ert_dir = os.path.abspath(os.path.dirname(args.config))
            os.chdir(ert_dir)
            # Changing current working directory means we need to update
            # the config file to be the base name of the original config
            args.config = os.path.basename(args.config)
            ert_config = ErtConfig.from_file(args.config)
            local_storage_set_ert_config(ert_config)
            ert = EnKFMain(ert_config)
        except ConfigValidationError as error:
            config_warnings = [
                w.message.info
                for w in all_warnings
                if w.category == ConfigWarning
                and not cast(ConfigWarning, w.message).info.is_deprecation
            ]
            deprecations = [
                w.message.info
                for w in all_warnings
                if w.category == ConfigWarning
                and cast(ConfigWarning, w.message).info.is_deprecation
            ]
            error_messages += error.errors
            logger.info("Error in config file shown in gui: '%s'", str(error))
            return (
                Suggestor(
                    error_messages,
                    config_warnings,
                    deprecations,
                    None,
                    (
                        plugin_manager.get_help_links()
                        if plugin_manager is not None
                        else {}
                    ),
                ),
                None,
                None,
                None,
            )
    config_warnings = [
        w.message.info
        for w in all_warnings
        if w.category == ConfigWarning
        and not cast(ConfigWarning, w.message).info.is_deprecation
    ]
    deprecations = [
        w.message.info
        for w in all_warnings
        if w.category == ConfigWarning
        and cast(ConfigWarning, w.message).info.is_deprecation
    ]
    for job in ert_config.forward_model_list:
        logger.info("Config contains forward model job %s", job.name)

    for wm in all_warnings:
        if wm.category != ConfigWarning:
            logger.warning(str(wm.message))
    for msg in deprecations:
        logger.info("Suggestion shown in gui '%s'", msg)
    for msg in config_warnings:
        logger.info("Warning shown in gui '%s'", msg)
    storage = open_storage(ert_config.ens_path, mode="w")
    _main_window = _setup_main_window(ert, args, log_handler, storage, plugin_manager)
    if deprecations or config_warnings:

        def continue_action():
            _main_window.show()
            _main_window.activateWindow()
            _main_window.raise_()
            _main_window.adjustSize()

        suggestor = Suggestor(
            error_messages,
            config_warnings,
            deprecations,
            continue_action,
            plugin_manager.get_help_links() if plugin_manager is not None else {},
        )
        suggestor.notifier = _main_window.notifier
        return (
            suggestor,
            ert_config.ens_path,
            ert_config.model_config.num_realizations,
            ert_config.ensemble_config.parameter_configuration,
        )
    else:
        return (
            _main_window,
            ert_config.ens_path,
            ert_config.model_config.num_realizations,
            ert_config.ensemble_config.parameter_configuration,
        )


def _check_locale():
    # There seems to be a setlocale() call deep down in the initialization of
    # QApplication, if the user has set the LC_NUMERIC environment variables to
    # a locale with decimalpoint different from "." the application will fail
    # hard quite quickly.
    current_locale = QLocale()
    decimal_point = str(current_locale.decimalPoint())
    if decimal_point != ".":
        msg = f"""You are using a locale with decimalpoint: '{decimal_point}'
the ert application is written with the assumption that '.' is  used as
decimalpoint, and chances are that something will break if you continue with
this locale. It is highly recommended that you set the decimalpoint to '.'
using one of the environment variables 'LANG', LC_ALL', or 'LC_NUMERIC' to
either the 'C' locale or alternatively a locale which uses '.' as
decimalpoint.\n"""  # noqa
        warnings.warn(msg, category=ConfigWarning, stacklevel=1)


def _clicked_help_button(menu_label: str, link: str):
    logger = logging.getLogger(__name__)
    logger.info(f"Pressed help button {menu_label}")
    webbrowser.open(link)


def _clicked_about_button(about_dialog):
    logger = logging.getLogger(__name__)
    logger.info("Pressed help button About")
    about_dialog.show()


def _setup_main_window(
    ert: EnKFMain,
    args: Namespace,
    log_handler: GUILogHandler,
    storage: Storage,
    plugin_manager: Optional[ErtPluginManager] = None,
) -> ErtMainWindow:
    # window reference must be kept until app.exec returns:
    facade = LibresFacade(ert)
    config_file = args.config
    config = ert.ert_config
    window = ErtMainWindow(config_file, plugin_manager)
    window.notifier.set_storage(storage)
    window.setWidget(SimulationPanel(ert, window.notifier, config_file))
    plugin_handler = PluginHandler(
        ert,
        window.notifier,
        [wfj for wfj in ert.ert_config.workflow_jobs.values() if wfj.is_plugin()],
        window,
    )

    window.addDock(
        "Configuration summary", SummaryPanel(ert), area=Qt.BottomDockWidgetArea
    )
    window.addTool(PlotTool(config_file, window))
    window.addTool(ExportTool(ert, window.notifier))
    window.addTool(WorkflowsTool(ert, window.notifier))
    window.addTool(
        ManageExperimentsTool(
            config, window.notifier, config.model_config.num_realizations
        )
    )
    window.addTool(PluginsTool(plugin_handler, window.notifier))
    window.addTool(RunAnalysisTool(ert, window.notifier))
    window.addTool(LoadResultsTool(facade, window.notifier))
    event_viewer = EventViewerTool(log_handler)
    window.addTool(event_viewer)
    window.close_signal.connect(event_viewer.close_wnd)
    window.adjustSize()
    return window
