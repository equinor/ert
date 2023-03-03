import functools
import logging
import os
import warnings
import webbrowser
from typing import Optional

from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from qtpy.QtCore import QLocale, QSize, Qt
from qtpy.QtWidgets import QApplication

from ert._c_wrappers.config import ConfigWarning
from ert._c_wrappers.config.config_parser import ConfigValidationError
from ert._c_wrappers.enkf import EnKFMain, ErtConfig
from ert.gui.about_dialog import AboutDialog
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
from ert.services import StorageService
from ert.shared.plugins.plugin_manager import ErtPluginManager
from ert.storage import EnsembleAccessor, StorageReader, open_storage


def run_gui(args: Namespace):
    app = QApplication([])  # Early so that QT is initialized before other imports
    app.setWindowIcon(resourceIcon("application/window_icon_cutout"))
    with add_gui_log_handler() as log_handler:
        window, ens_path, ensemble_size = _start_initial_gui_window(args, log_handler)

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

        mode = "r" if args.read_only else "w"
        with StorageService.init_service(
            ert_config=args.config, project=os.path.abspath(ens_path)
        ), open_storage(ens_path, mode=mode) as storage:
            if hasattr(window, "notifier"):
                window.notifier.set_storage(storage)
                window.notifier.set_current_case(
                    _get_or_create_default_case(storage, ensemble_size)
                )
            return show_window()


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
        return (
            _setup_suggester(error_messages, warning_messages, suggestions),
            None,
            None,
        )

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
            ert_config.model_config.num_realizations,
        )
    else:
        return (
            _main_window,
            ert_config.ens_path,
            ert_config.model_config.num_realizations,
        )


def _get_or_create_default_case(
    storage: StorageReader, ensemble_size: int
) -> Optional[EnsembleAccessor]:
    try:
        storage_accessor = storage.to_accessor()
        try:
            return storage_accessor.get_ensemble_by_name("default")
        except KeyError:
            return storage_accessor.create_experiment().create_ensemble(
                name="default", ensemble_size=ensemble_size
            )
    except TypeError:
        return None


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


def _clicked_help_button(menu_label: str, link: str):
    logger = logging.getLogger(__name__)
    logger.info(f"Pressed help button {menu_label}")
    webbrowser.open(link)


def _clicked_about_button(about_dialog):
    logger = logging.getLogger(__name__)
    logger.info("Pressed help button About")
    about_dialog.show()


def _setup_suggester(errors, warning_msgs, suggestions, ert_window=None):
    container = QWidget()
    if ert_window is not None:
        container.notifier = ert_window.notifier
    container.setWindowTitle("Some problems detected")
    container_layout = QVBoxLayout()

    help_button_frame = QFrame()
    help_buttons_layout = QHBoxLayout()
    help_button_frame.setLayout(help_buttons_layout)

    button_size = QSize(-1, -1)
    helpbuttons = []

    help_label = QLabel("Help:")
    help_buttons_layout.addWidget(help_label)
    pm = ErtPluginManager()
    help_links = pm.get_help_links()

    for menu_label, link in help_links.items():
        button = QPushButton(menu_label)
        button.setObjectName(menu_label)
        button.clicked.connect(
            functools.partial(_clicked_help_button, menu_label, link)
        )
        helpbuttons.append(button)
        help_buttons_layout.addWidget(button)

    about_button = QPushButton("About")
    about_button.setObjectName("about_button")
    helpbuttons.append(about_button)
    help_buttons_layout.addWidget(about_button)

    diag = AboutDialog(container)
    about_button.clicked.connect(lambda: _clicked_about_button(diag))

    for b in helpbuttons:
        b.adjustSize()
        if b.size().width() > button_size.width():
            button_size = b.size()

    for b in helpbuttons:
        b.setMinimumSize(button_size)

    help_buttons_layout.insertStretch(-1, -1)

    container_layout.addWidget(help_button_frame)

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
    buttons_layout.insertStretch(-1, -1)
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
    window = ErtMainWindow(config_file)
    window.setWidget(SimulationPanel(ert, window.notifier, config_file))
    plugin_handler = PluginHandler(
        ert,
        window.notifier,
        [wfj for wfj in ert.resConfig().workflow_jobs.values() if wfj.isPlugin()],
        window,
    )

    window.addDock(
        "Configuration summary", SummaryPanel(ert), area=Qt.BottomDockWidgetArea
    )
    window.addTool(PlotTool(config_file, window))
    window.addTool(ExportTool(ert))
    window.addTool(WorkflowsTool(ert, window.notifier))
    window.addTool(ManageCasesTool(ert, window.notifier))
    window.addTool(PluginsTool(plugin_handler, window.notifier))
    window.addTool(RunAnalysisTool(ert, window.notifier))
    window.addTool(LoadResultsTool(facade, window.notifier))
    event_viewer = EventViewerTool(log_handler)
    window.addTool(event_viewer)
    window.close_signal.connect(event_viewer.close_wnd)
    window.adjustSize()
    return window
