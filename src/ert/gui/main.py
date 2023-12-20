import functools
import logging
import os
import warnings
import webbrowser
from signal import SIG_DFL, SIGINT, signal
from typing import Optional, cast

import pkg_resources
from PyQt5.QtGui import QCursor, QIcon
from qtpy.QtCore import QDir, QLocale, Qt
from qtpy.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ert.config import ConfigValidationError, ConfigWarning, ErtConfig
from ert.enkf_main import EnKFMain
from ert.gui.about_dialog import AboutDialog
from ert.gui.ertwidgets import SuggestorMessage, SummaryPanel
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
from ert.storage import open_storage
from ert.storage.local_storage import local_storage_set_ert_config


def run_gui(args: Namespace, plugin_manager: Optional[ErtPluginManager] = None):
    # Replace Python's exception handler for SIGINT with the system default.
    #
    # Python's SIGINT handler is the one that raises KeyboardInterrupt. This is
    # okay normally (if a bit ugly), but when control is given to Qt this
    # exception handler will either get deadlocked because Python never gets
    # control back, or gets eaten by Qt because it ignores exceptions that
    # happen in Qt slots.
    signal(SIGINT, SIG_DFL)

    QDir.addSearchPath(
        "img", pkg_resources.resource_filename("ert.gui", "resources/gui/img")
    )

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

        mode = "r" if args.read_only else "w"
        with StorageService.init_service(
            ert_config=args.config, project=os.path.abspath(ens_path)
        ), open_storage(ens_path, mode=mode) as storage:
            if hasattr(window, "notifier"):
                window.notifier.set_storage(storage)
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
                _setup_suggester(
                    error_messages,
                    config_warnings,
                    deprecations,
                    plugin_manager=plugin_manager,
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
    _main_window = _setup_main_window(ert, args, log_handler, plugin_manager)
    if deprecations or config_warnings:
        return (
            _setup_suggester(
                error_messages,
                config_warnings,
                deprecations,
                _main_window,
                plugin_manager=plugin_manager,
            ),
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


BUTTON_STYLE = """
QPushButton {
    background-color: #007079;
    color: white;
    border-radius: 4px;
    border: 2px solid #045e8d;
    height: 36px;
    padding: 0px 16px 0px 16px;
}
QPushButton:hover {
    background: #004f55;
};
"""

LINK_STYLE = """
QPushButton {
    color: #045e8d;
    border: 0px solid white;
    margin-left: 34px;
    height: 36px;
    padding: 0px 12px 0px 12px;
    text-decoration: underline;
    text-align: left;
    font-size: 16px;
    padding: 0px;
}
"""

DISABLED_BUTTON_STYLE = """
    background-color: #eaeaea;
    color: #bebebe;
    border-radius: 4px;
    border: 2px solid #eaeaea;
    height: 36px;
    padding: 0px 16px 0px 16px;
"""

SECONDARY_BUTTON_STYLE = """
QPushButton {
    background-color: #f7f7f7;
    color: #007079;
    border-radius: 4px;
    border: 2px solid #007079;
    height: 36px;
    padding: 0px 16px 0px 16px;
}
QPushButton:hover {
    background-color: #deedee;
};
"""


def _setup_suggester(
    errors,
    warning_msgs,
    suggestions,
    ert_window=None,
    plugin_manager: Optional[ErtPluginManager] = None,
):
    container = QWidget()
    container.setLayout(QVBoxLayout())
    data_widget = QWidget()
    container.layout().addWidget(
        QLabel(
            """\
            <p style="font-size: 28px;">Some problems detected</p>
            <p> The following problems were detected while reading
            the ert configuration file. </p>
    """
        )
    )
    container.setWindowTitle("ERT")
    container.layout().addWidget(data_widget)
    container.resize(1440, 1024)
    container.setStyleSheet("background-color: #f7f7f7;")
    container.layout().setContentsMargins(32, 47, 32, 16)
    container.layout().setSpacing(32)

    if ert_window is not None:
        data_widget.notifier = ert_window.notifier
    data_widget.setLayout(QHBoxLayout())
    data_widget.layout().setSpacing(16)
    data_widget.layout().setContentsMargins(0, 0, 0, 0)

    help_button_frame = QFrame()
    help_button_frame.setContentsMargins(0, 0, 0, 0)
    help_button_frame.setStyleSheet(
        "background-color: #eaeaea; border-radius: 4px; border: 2px solid #dcdcdc"
    )
    help_button_frame.setMinimumWidth(388)
    help_button_frame.setMaximumWidth(388)
    help_buttons_layout = QVBoxLayout()
    help_buttons_layout.setContentsMargins(0, 30, 20, 20)
    help_button_frame.setLayout(help_buttons_layout)

    help_links = plugin_manager.get_help_links() if plugin_manager else {}

    help_header = QLabel("Helpful links")
    help_header.setContentsMargins(0, 0, 0, 0)
    help_header.setStyleSheet(
        "font-size: 24px; color: #045e8d; border: none; margin-left: 30px;"
    )
    help_buttons_layout.addWidget(help_header, alignment=Qt.AlignTop)

    separator = QFrame()
    separator.setFrameShape(QFrame.HLine)
    separator.setStyleSheet("color: #dcdcdc;")
    separator.setFixedWidth(388)
    help_buttons_layout.addWidget(separator)

    for menu_label, link in help_links.items():
        button = QPushButton(menu_label)
        button.setStyleSheet(LINK_STYLE)
        button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        button.setObjectName(menu_label)
        button.clicked.connect(
            functools.partial(_clicked_help_button, menu_label, link)
        )
        help_buttons_layout.addWidget(button)

    about_button = QPushButton("About")
    about_button.setStyleSheet(LINK_STYLE)
    about_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
    about_button.setObjectName("about_button")
    help_buttons_layout.addWidget(about_button)
    help_buttons_layout.addStretch(-1)

    diag = AboutDialog(data_widget)
    about_button.clicked.connect(lambda: _clicked_about_button(diag))

    suggest_msgs = QWidget()
    suggest_msgs.setContentsMargins(0, 0, 16, 0)
    buttons = QWidget()
    suggest_layout = QGridLayout()
    suggest_layout.setContentsMargins(0, 0, 0, 0)
    suggest_layout.setColumnMinimumWidth(0, 450)
    suggest_layout.setSpacing(24)
    buttons_layout = QHBoxLayout()

    column = 0
    row = 0
    num = 0
    for msg in errors:
        suggest_layout.addWidget(SuggestorMessage.error_msg(msg), row, column)
        if column:
            row += 1
        column = (column + 1) % 2
        num += 1
    for msg in warning_msgs:
        suggest_layout.addWidget(SuggestorMessage.warning_msg(msg), row, column)
        if column:
            row += 1
        column = (column + 1) % 2
        num += 1
    for msg in suggestions:
        suggest_layout.addWidget(SuggestorMessage.deprecation_msg(msg), row, column)
        if column:
            row += 1
        column = (column + 1) % 2
        num += 1
    suggest_layout.setRowStretch(row + 1, 1)
    width = 1440
    height = 1024
    if num <= 1:
        width -= 450
    else:
        suggest_layout.setColumnMinimumWidth(1, 450)
        suggest_layout.setColumnStretch(2, 1)
    if row < 4:
        height -= (4 - (row + column)) * 150
    container.resize(width, height)

    suggest_msgs.setLayout(suggest_layout)
    scroll = QScrollArea()
    scroll.setStyleSheet(
        """
        QScrollArea {
            border: none;
            width: 128px;
        }
        QScrollBar {
            border: none;
            background-color: #f7f7f7;
            width: 10px;
        }
        QScrollBar::handle {
            border: none;
            background-color: #dcdcdc;
            border-radius: 4px;
        }
        QScrollBar::sub-line:vertical, QScrollBar::add-line:vertical {
            background: none;
        }
    """
    )
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidgetResizable(True)
    scroll.setWidget(suggest_msgs)
    scroll.setContentsMargins(0, 0, 0, 0)

    def run_pressed():
        ert_window.show()
        ert_window.activateWindow()
        ert_window.raise_()
        ert_window.adjustSize()
        container.close()

    run = QPushButton("Open ERT")
    give_up = QPushButton("Cancel")
    if ert_window is None:
        run.setStyleSheet(DISABLED_BUTTON_STYLE)
        run.setEnabled(False)
        give_up.setStyleSheet(BUTTON_STYLE)
    else:
        run.setStyleSheet(BUTTON_STYLE)
        run.setEnabled(True)
        give_up.setStyleSheet(SECONDARY_BUTTON_STYLE)

    run.setObjectName("run_ert_button")
    run.pressed.connect(run_pressed)
    give_up.pressed.connect(container.close)

    buttons_layout.insertStretch(-1, -1)
    buttons_layout.setContentsMargins(0, 24, 0, 0)
    buttons_layout.addWidget(run)
    buttons_layout.addWidget(give_up)

    buttons.setLayout(buttons_layout)
    problem_area = QWidget()
    problem_area.setLayout(QVBoxLayout())
    problem_area.setContentsMargins(0, 0, 0, 0)
    problem_area.layout().setContentsMargins(0, 0, 0, 0)
    problem_area.layout().addWidget(scroll)
    problem_area.layout().addWidget(buttons)
    data_widget.layout().addWidget(problem_area)
    data_widget.layout().addWidget(help_button_frame)
    return container


def _setup_main_window(
    ert: EnKFMain,
    args: Namespace,
    log_handler: GUILogHandler,
    plugin_manager: Optional[ErtPluginManager] = None,
):
    # window reference must be kept until app.exec returns:
    facade = LibresFacade(ert)
    config_file = args.config
    config = ert.ert_config
    window = ErtMainWindow(config_file, plugin_manager)
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
        ManageCasesTool(config, window.notifier, config.model_config.num_realizations)
    )
    window.addTool(PluginsTool(plugin_handler, window.notifier))
    window.addTool(RunAnalysisTool(ert, window.notifier))
    window.addTool(LoadResultsTool(facade, window.notifier))
    event_viewer = EventViewerTool(log_handler)
    window.addTool(event_viewer)
    window.close_signal.connect(event_viewer.close_wnd)
    window.adjustSize()
    return window
