import logging
import os
import warnings
from pathlib import Path

import filelock
from PyQt5.QtWidgets import QHBoxLayout, QPushButton, QScrollArea, QVBoxLayout, QWidget
from qtpy.QtCore import QLocale, Qt
from qtpy.QtWidgets import QApplication

from ert._c_wrappers.config import ConfigWarning
from ert._c_wrappers.config.config_parser import ConfigValidationError
from ert._c_wrappers.enkf import EnKFMain, ErtConfig
from ert.cli.main import ErtTimeoutError
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import SuggestorMessage, SummaryPanel, resourceIcon
from ert.gui.main_window import ErtMainWindow
from ert.gui.simulation import SimulationPanel
from ert.gui.tools.event_viewer import (
    EventViewerTool,
    GUILogHandler,
    add_gui_log_handler,
)
from ert.gui.tools.export import ExportTool
from ert.gui.tools.load_results import LoadResultsTool
from ert.gui.tools.manage_cases import ManageCasesTool
from ert.gui.tools.plot import PlotTool
from ert.gui.tools.plugins import PluginHandler, PluginsTool
from ert.gui.tools.run_analysis import RunAnalysisTool
from ert.gui.tools.workflows import WorkflowsTool
from ert.libres_facade import LibresFacade
from ert.namespace import Namespace
from ert.services import Storage


def run_gui(args: Namespace):
    app = QApplication([])  # Early so that QT is initialized before other imports
    app.setWindowIcon(resourceIcon("application/window_icon_cutout"))
    with add_gui_log_handler() as log_handler:
        window, ens_path = _start_initial_gui_window(args, log_handler)

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

        storage_lock = filelock.FileLock(
            Path(ens_path) / (Path(ens_path).stem + ".lock")
        )
        try:
            storage_lock.acquire(timeout=5)
            with Storage.init_service(
                ert_config=args.config,
                project=os.path.abspath(ens_path),
            ):
                return show_window()
        except filelock.Timeout:
            raise ErtTimeoutError(
                f"Not able to acquire lock for: {ens_path}. You may already be running"
                f" ert, or another user is using the same ENSPATH."
            )
        finally:
            if storage_lock.is_locked:
                storage_lock.release()
                os.remove(storage_lock.lock_file)


def _start_initial_gui_window(args, log_handler):
    # Create logger inside function to make sure all handlers have been added to
    # the root-logger.
    logger = logging.getLogger(__name__)
    suggestions = []
    error_messages = []
    warning_msgs = []
    ert_config = None
    try:
        with warnings.catch_warnings(record=True) as warning_messages:
            _check_locale()
            suggestions += ErtConfig.make_suggestion_list(args.config)
            ert_config = ErtConfig.from_file(args.config)
        os.chdir(ert_config.config_path)
        # Changing current working directory means we need to update the config file to
        # be the base name of the original config
        args.config = os.path.basename(args.config)
        ert = EnKFMain(ert_config)
        warning_msgs = warning_messages

    except ConfigValidationError as error:
        error_messages.append(str(error))
        logger.info("Error in config file shown in gui: '%s'", str(error))
        return _setup_suggester(error_messages, warning_messages, suggestions), None

    for job in ert_config.forward_model_list:
        logger.info("Config contains forward model job %s", job.name)

    for wm in warning_msgs:
        if wm.category != ConfigWarning:
            logger.warning(str(wm.message))
    for msg in suggestions:
        logger.info("Suggestion shown in gui '%s'", msg)
    _main_window = _setup_main_window(ert, args, log_handler)
    if suggestions or warning_msgs:
        return (
            _setup_suggester(error_messages, warning_msgs, suggestions, _main_window),
            ert_config.ens_path,
        )
    else:
        return _main_window, ert_config.ens_path


def _check_locale():
    # There seems to be a setlocale() call deep down in the initialization of
    # QApplication, if the user has set the LC_NUMERIC environment variables to
    # a locale with decimalpoint different from "." the application will fail
    # hard quite quickly.
    current_locale = QLocale()
    decimal_point = str(current_locale.decimalPoint())
    if decimal_point != ".":
        msg = f"""
** WARNING: You are using a locale with decimalpoint: '{decimal_point}' - the ert application is
written with the assumption that '.' is  used as decimalpoint, and chances
are that something will break if you continue with this locale. It is highly
recommended that you set the decimalpoint to '.' using one of the environment
variables 'LANG', LC_ALL', or 'LC_NUMERIC' to either the 'C' locale or
alternatively a locale which uses '.' as decimalpoint.\n"""  # noqa
        warnings.warn(msg, category=UserWarning)


def _setup_suggester(errors, warning_msgs, suggestions, ert_window=None):
    container = QWidget()
    container.setWindowTitle("Some problems detected")
    container_layout = QVBoxLayout()

    suggest_msgs = QWidget()
    buttons = QWidget()
    suggest_layout = QVBoxLayout()
    buttons_layout = QHBoxLayout()
    text = ""
    for msg in errors:
        text += msg + "\n"
        suggest_layout.addWidget(SuggestorMessage.error_msg(msg))
    for msg in warning_msgs:
        msg = str(msg.message)
        text += msg + "\n"
        suggest_layout.addWidget(SuggestorMessage.warning_msg(msg))
    for msg in suggestions:
        text += msg + "\n"
        suggest_layout.addWidget(SuggestorMessage.deprecation_msg(msg))

    suggest_layout.addStretch()
    suggest_msgs.setLayout(suggest_layout)
    scroll = QScrollArea()
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setWidgetResizable(True)
    scroll.setWidget(suggest_msgs)

    def copy_text():
        QApplication.clipboard().setText(text)

    def run_pressed():
        ert_window.show()
        ert_window.activateWindow()
        ert_window.raise_()
        container.close()

    run = QPushButton("Run ert")
    give_up = QPushButton("Exit")
    copy = QPushButton("Copy messages")

    run.setObjectName("run_ert_button")
    run.setEnabled(ert_window is not None)
    run.pressed.connect(run_pressed)
    copy.pressed.connect(copy_text)
    give_up.pressed.connect(container.close)

    buttons_layout.addWidget(copy)
    buttons_layout.addWidget(run)
    buttons_layout.addWidget(give_up)

    buttons.setLayout(buttons_layout)
    container_layout.addWidget(scroll)
    container_layout.addWidget(buttons)
    container.setLayout(container_layout)
    container.resize(800, 600)
    return container


def _setup_main_window(
    ert: EnKFMain,
    args: Namespace,
    log_handler: GUILogHandler,
):
    # window reference must be kept until app.exec returns:
    facade = LibresFacade(ert)
    config_file = args.config
    notifier = ErtNotifier(config_file)
    window = ErtMainWindow(config_file)
    window.setWidget(SimulationPanel(ert, notifier, config_file))
    plugin_handler = PluginHandler(
        ert,
        [wfj for wfj in ert.resConfig().workflow_jobs.values() if wfj.isPlugin()],
        window,
    )

    window.addDock(
        "Configuration summary", SummaryPanel(ert), area=Qt.BottomDockWidgetArea
    )
    window.addTool(PlotTool(config_file, window))
    window.addTool(ExportTool(ert))
    window.addTool(WorkflowsTool(ert, notifier))
    window.addTool(ManageCasesTool(ert, notifier))
    window.addTool(PluginsTool(plugin_handler, notifier))
    window.addTool(RunAnalysisTool(ert, notifier))
    window.addTool(LoadResultsTool(facade))
    event_viewer = EventViewerTool(log_handler)
    window.addTool(event_viewer)
    window.close_signal.connect(event_viewer.close_wnd)
    window.adjustSize()
    return window
