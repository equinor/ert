from functools import partial

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QAction,
    QLabel,
    QMainWindow,
    QMenu,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from everest import __version__ as everest_version
from everest.config import EverestConfig
from ieverest.config_widget import ConfigWidget
from ieverest.monitor_widget import MonitorWidget
from ieverest.utils import load_ui


class MainWindow(QMainWindow):
    """The main window of the Everest GUI

    It contains the usual stuff (menus, statusbar, etc) and 3 tabs: one
    for setting up a configuration, one for starting and monitoring the
    execution, one for processing the results.
    """

    export_requested = Signal()
    open_requested = Signal(str)
    quit_requested = Signal()

    def __init__(self, parent=None):
        self._config = EverestConfig.with_defaults(**{})
        self._actions = {}
        self._recent_files = []

        super(MainWindow, self).__init__(parent)
        load_ui("main_window.ui", self)
        self.setWindowTitle("IEverest")

        self.tabs = QTabWidget()
        self.monitor_gui = MonitorWidget(self.tabs)
        self.config_gui = ConfigWidget(self.tabs)
        self.tabs.addTab(self.monitor_gui, "Optimization")
        self.tabs.addTab(self.config_gui, "Configuration")
        self.tabs.setVisible(False)  # will be shown when a config is set

        self._init_actions()
        self._init_menus()

        self._set_startup_ui()  # must be after init_actions and init_menus

    def _init_actions(self):
        self.register_action(
            "open",
            "&Open...",
            self,
            icon=QIcon.fromTheme("document-open"),
            shortcut=Qt.CTRL + Qt.Key_O,
            on_triggered=partial(self.open_requested.emit, ""),
        )
        self.register_action(
            "export", "&Export results...", self, on_triggered=self.export_requested
        )
        self.register_action(
            "quit",
            "&Quit",
            self,
            icon=QIcon.fromTheme("application-exit"),
            on_triggered=self.quit_requested,
        )

    def _init_menus(self):
        file_menu = self.menubar.findChildren(QMenu, "menu_file")
        assert len(file_menu) == 1
        file_menu = file_menu[0]
        recent_menu = QMenu("&Recent", file_menu)
        recent_menu.setObjectName("menu_recent_files")
        file_menu.addAction(self.actions["open"])
        file_menu.addMenu(recent_menu)
        file_menu.addSeparator()
        file_menu.addAction(self.actions["export"])
        file_menu.addSeparator()
        file_menu.addAction(self.actions["quit"])

        help_menu = self.menuBar().addMenu("Help")
        help_menu.setObjectName("help_menu")
        about_action = help_menu.addAction("About")
        about_action.setObjectName("about_action")
        about_action.triggered.connect(lambda x: self.show_about())

        self.about_widget = QWidget(self)
        self.about_widget.setWindowFlags(Qt.Window)
        self.about_widget.setObjectName("about_widget")

        layout = QVBoxLayout(self.about_widget)
        layout.addWidget(QLabel("<b>Everest version:<\b>"))
        layout.addWidget(QLabel("{version}".format(version=everest_version)))

        github_label = QLabel(
            '<a href="https://github.com/equinor/everest">Everest GitHub</a>', self
        )
        github_label.setOpenExternalLinks(True)
        layout.addWidget(github_label)
        layout.addStretch(30)

        close_button = QPushButton("Close", self.about_widget)
        close_button.setObjectName("button_close_about")
        layout.addWidget(close_button)
        close_button.clicked.connect(lambda x: self.hide_about())

    def hide_about(self):
        self.about_widget.setVisible(False)
        self.about_widget.setWindowModality(Qt.WindowModality.NonModal)

    def show_about(self):
        self.about_widget.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.about_widget.setVisible(True)
        self.about_widget.setWindowTitle("About Everest")

    def _set_startup_ui(self):
        self._startup_gui = load_ui("startup_widget.ui")
        self._startup_gui.open_btn.clicked.connect(self.actions["open"].triggered.emit)

        recent_menu = self.menubar.findChildren(QMenu, "menu_recent_files")
        assert len(recent_menu) == 1
        recent_menu = recent_menu[0]
        self._startup_gui.recent_btn.setMenu(recent_menu)

        self.setCentralWidget(self._startup_gui)

    @property
    def actions(self):
        return self._actions

    @property
    def recent_files(self):
        return self._recent_files

    @recent_files.setter
    def recent_files(self, files):
        self._recent_files = files
        menu = self.menubar.findChildren(QMenu, "menu_recent_files")
        assert len(menu) == 1
        menu = menu[0]
        menu.clear()
        self._startup_gui.recent_btn.setEnabled(bool(files))
        for rf in files:
            act = QAction(rf, menu)
            act.triggered.connect(partial(self.open_requested.emit, rf))
            menu.addAction(act)

    def register_action(
        self,
        action_id,
        text,
        parent,
        icon=None,
        enabled=True,
        checkable=False,
        checked=False,
        shortcut=None,
        shortcutcontext=None,
        on_hovered=None,
        on_toggled=None,
        on_triggered=None,
    ):
        act = QAction(icon, text, parent) if icon is not None else QAction(text, parent)
        act.setEnabled(enabled)
        act.setCheckable(checkable)
        act.setChecked(checked)
        if shortcut is not None:
            act.setShortcut(shortcut)
        if shortcutcontext is not None:
            act.setShortcutContext(shortcutcontext)
        if on_hovered is not None:
            act.hovered.connect(on_hovered)
        if on_toggled is not None:
            act.toggled.connect(on_toggled)
        if on_triggered is not None:
            act.triggered.connect(on_triggered)
        self._actions[action_id] = act
        return act

    def set_config(self, config: EverestConfig):
        # On first config, enable the "main" ui
        if self.centralWidget() is not self.tabs:
            self.tabs.setVisible(True)
            self.setCentralWidget(self.tabs)

        self._config = config
        self.config_gui.set_config(config)

        self.monitor_gui.set_config(config)
        self.monitor_gui.create_objective_fn_widgets_from_config()

        self.setWindowTitle(f"IEverest - {self._config.config_path}")

    def get_config(self) -> EverestConfig:
        return self._config

    # --- Overridden methods ---

    def closeEvent(self, event):
        # event triggered when the user click on the X on the top right
        # notify that quitting has been requested and ignore the event
        self.quit_requested.emit()
        event.ignore()
