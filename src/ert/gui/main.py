from __future__ import annotations

import logging
import os
import sys
import warnings
import webbrowser
from signal import SIG_DFL, SIGINT, signal
from typing import Optional, Tuple, cast

if sys.version_info >= (3, 9):
    from importlib.resources import files
else:
    from importlib_resources import files

from collections import Counter

from qtpy.QtCore import QDir, Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QApplication, QWidget

from ert.config import ConfigValidationError, ConfigWarning, ErrorInfo, ErtConfig
from ert.gui.main_window import ErtMainWindow
from ert.gui.simulation import ExperimentPanel
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
from ert.gui.tools.workflows import WorkflowsTool
from ert.libres_facade import LibresFacade
from ert.namespace import Namespace
from ert.plugins import ErtPluginManager
from ert.services import StorageService
from ert.storage import ErtStorageException, Storage, open_storage
from ert.storage.local_storage import local_storage_set_ert_config

from .suggestor import Suggestor
from .summarypanel import SummaryPanel


def run_gui(args: Namespace, plugin_manager: Optional[ErtPluginManager] = None) -> int:
    # Replace Python's exception handler for SIGINT with the system default.
    #
    # Python's SIGINT handler is the one that raises KeyboardInterrupt. This is
    # okay normally (if a bit ugly), but when control is given to Qt this
    # exception handler will either get deadlocked because Python never gets
    # control back, or gets eaten by Qt because it ignores exceptions that
    # happen in Qt slots.
    signal(SIGINT, SIG_DFL)

    QDir.addSearchPath("img", str(files("ert.gui").joinpath("resources/gui/img")))

    app = QApplication(["ert"])  # Early so that QT is initialized before other imports
    app.setWindowIcon(QIcon("img:ert_icon.svg"))

    with add_gui_log_handler() as log_handler:
        window, ens_path = _start_initial_gui_window(args, log_handler, plugin_manager)

        def show_window() -> int:
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
    return -1


def _start_initial_gui_window(
    args: Namespace,
    log_handler: GUILogHandler,
    plugin_manager: Optional[ErtPluginManager] = None,
) -> Tuple[QWidget, Optional[str]]:
    # Create logger inside function to make sure all handlers have been added to
    # the root-logger.
    logger = logging.getLogger(__name__)
    error_messages = []
    config_warnings = []
    deprecations = []
    ert_config = None

    with warnings.catch_warnings(record=True) as all_warnings:
        try:
            ert_dir = os.path.abspath(os.path.dirname(args.config))
            os.chdir(ert_dir)
            # Changing current working directory means we need to update
            # the config file to be the base name of the original config
            args.config = os.path.basename(args.config)

            ert_config = ErtConfig.with_plugins().from_file(args.config)

            local_storage_set_ert_config(ert_config)
        except ConfigValidationError as error:
            config_warnings = [
                cast(ConfigWarning, w.message).info
                for w in all_warnings
                if w.category == ConfigWarning
                and not cast(ConfigWarning, w.message).info.is_deprecation
            ]
            deprecations = [
                cast(ConfigWarning, w.message).info
                for w in all_warnings
                if w.category == ConfigWarning
                and cast(ConfigWarning, w.message).info.is_deprecation
            ]
            error_messages += error.errors
        if ert_config is not None:
            try:
                storage = open_storage(ert_config.ens_path, mode="w")
            except ErtStorageException as err:
                error_messages.append(
                    ErrorInfo(f"Error opening storage in ENSPATH: {err}").set_context(
                        ert_config.ens_path
                    )
                )
        if error_messages:
            logger.info(f"Error in config file shown in gui: {error_messages}")
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
            )
    assert ert_config is not None
    config_warnings = [
        cast(ConfigWarning, w.message).info
        for w in all_warnings
        if w.category == ConfigWarning
        and not cast(ConfigWarning, w.message).info.is_deprecation
    ]
    deprecations = [
        cast(ConfigWarning, w.message).info
        for w in all_warnings
        if w.category == ConfigWarning
        and cast(ConfigWarning, w.message).info.is_deprecation
    ]
    counter_fm_steps = Counter(fms.name for fms in ert_config.forward_model_steps)

    for fm_step_name, count in counter_fm_steps.items():
        logger.info(
            f"Config contains forward model step {fm_step_name} {count} time(s)",
        )

    for wm in all_warnings:
        if wm.category != ConfigWarning:
            logger.warning(str(wm.message))
    for msg in deprecations:
        logger.info(f"Suggestion shown in gui '{msg}'")
    for msg in config_warnings:
        logger.info(f"Warning shown in gui '{msg}'")
    _main_window = _setup_main_window(
        ert_config, args, log_handler, storage, plugin_manager
    )
    if deprecations or config_warnings:

        def continue_action() -> None:
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
        )
    else:
        return (
            _main_window,
            ert_config.ens_path,
        )


def _clicked_help_button(menu_label: str, link: str) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"Pressed help button {menu_label}")
    webbrowser.open(link)


def _clicked_about_button(about_dialog: QWidget) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Pressed help button About")
    about_dialog.show()


def _setup_main_window(
    config: ErtConfig,
    args: Namespace,
    log_handler: GUILogHandler,
    storage: Storage,
    plugin_manager: Optional[ErtPluginManager] = None,
) -> ErtMainWindow:
    # window reference must be kept until app.exec returns:
    facade = LibresFacade(config)
    config_file = args.config
    window = ErtMainWindow(config_file, plugin_manager)
    window.notifier.set_storage(storage)
    window.setWidget(
        ExperimentPanel(
            config, window.notifier, config_file, facade.get_ensemble_size()
        )
    )

    plugin_handler = PluginHandler(
        window.notifier,
        [wfj for wfj in config.workflow_jobs.values() if wfj.is_plugin()],
        window,
    )

    window.addDock(
        "Configuration summary",
        SummaryPanel(config),
        area=Qt.DockWidgetArea.BottomDockWidgetArea,
    )
    window.addTool(PlotTool(config_file, window))
    window.addTool(ExportTool(config, window.notifier))
    window.addTool(WorkflowsTool(config, window.notifier))
    window.addTool(
        ManageExperimentsTool(
            config, window.notifier, config.model_config.num_realizations
        )
    )
    window.addTool(PluginsTool(plugin_handler, window.notifier, config))
    window.addTool(LoadResultsTool(facade, window.notifier))
    event_viewer = EventViewerTool(log_handler, config_file)
    window.addTool(event_viewer)
    window.close_signal.connect(event_viewer.close_wnd)
    window.adjustSize()
    return window
