import logging
import os
import sys
from pathlib import Path

import filelock
from PyQt5.QtWidgets import QHBoxLayout, QPushButton, QTextEdit, QVBoxLayout, QWidget
from qtpy.QtCore import QLocale, Qt
from qtpy.QtWidgets import QApplication

from ert._c_wrappers.enkf import EnKFMain, ResConfig
from ert.cli.main import ErtTimeoutError
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import SummaryPanel, resourceIcon
from ert.gui.main_window import GertMainWindow
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
                res_config=args.config,
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
    messages = []
    res_config = None
    try:
        res_config = ResConfig(args.config)
        messages += ResConfig.make_suggestion_list(args.config)
    except Exception as error:
        messages.append(str(error))
        return _setup_suggester(messages, args, log_handler), None

    # Create logger inside function to make sure all handlers have been added to
    # the root-logger.
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging forward model jobs",
        extra={"workflow_jobs": str(res_config.forward_model.job_name_list())},
    )
    os.chdir(res_config.config_path)
    # Changing current working directory means we need to update the config file to
    # be the base name of the original config
    args.config = os.path.basename(args.config)
    try:
        ert = EnKFMain(res_config)
    except Exception as error:
        messages.append(str(error))
        return _setup_suggester(messages, args, log_handler), None

    locale_msg = _check_locale()
    if locale_msg is not None:
        messages.append(locale_msg)
    if messages:
        return (
            _setup_suggester(
                messages, args, log_handler, _setup_main_window(ert, args, log_handler)
            ),
            res_config.ens_path,
        )
    else:
        return _setup_main_window(ert, args, log_handler), res_config.ens_path


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

        sys.stderr.write(msg)
        return msg
    else:
        return None


def _setup_suggester(suggestions, args, log_handler, ert_window=None):
    suggest = QWidget()
    layout = QVBoxLayout()
    suggest.setWindowTitle("Some problems detected")
    lines = QTextEdit()
    lines.setReadOnly(True)
    text = "\n".join(suggestions)
    lines.setPlainText(text)

    buttons = QHBoxLayout()
    layout.addWidget(lines)

    copy = QPushButton("Copy messages")

    def copy_text():
        QApplication.clipboard().setText(text)

    copy.pressed.connect(copy_text)

    run = QPushButton("Run ert")
    run.setObjectName("run_ert_button")
    run.setEnabled(ert_window is not None)

    def run_pressed():
        ert_window.show()
        ert_window.activateWindow()
        ert_window.raise_()
        suggest.close()

    run.pressed.connect(run_pressed)
    give_up = QPushButton("Exit")

    give_up.pressed.connect(suggest.close)
    buttons.addWidget(copy)
    buttons.addWidget(run)
    buttons.addWidget(give_up)

    layout.addLayout(buttons)
    suggest.setLayout(layout)
    suggest.resize(800, 600)
    return suggest


def _setup_main_window(
    ert: EnKFMain,
    args: Namespace,
    log_handler: GUILogHandler,
):
    # window reference must be kept until app.exec returns:
    facade = LibresFacade(ert)
    config_file = args.config
    notifier = ErtNotifier(config_file)
    window = GertMainWindow(config_file)
    window.setWidget(SimulationPanel(ert, notifier, config_file))
    plugin_handler = PluginHandler(ert, ert.getWorkflowList().getPluginJobs(), window)

    window.addDock(
        "Configuration summary", SummaryPanel(ert), area=Qt.BottomDockWidgetArea
    )
    window.addTool(PlotTool(config_file))
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
