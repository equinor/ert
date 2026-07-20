from __future__ import annotations

from typing import override

from PyQt6.QtCore import QCoreApplication, QEvent, QObject, Qt
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtGui import QAction, QActionGroup, QIcon, QMouseEvent
from PyQt6.QtWidgets import QMenu, QToolBar, QToolButton, QWidget

from .icon_utils import load_icon

START_EXPERIMENT = "Start experiment"
CREATE_PLOT = "Create plot"
MANAGE_EXPERIMENTS = "Manage experiments"
EXPERIMENT_STATUS = "Experiment status"

NAVIGATION_ENTRIES: tuple[tuple[str, str], ...] = (
    (START_EXPERIMENT, "library_add.svg"),
    (CREATE_PLOT, "timeline.svg"),
    (MANAGE_EXPERIMENTS, "build_wrench.svg"),
    (EXPERIMENT_STATUS, "in_progress.svg"),
)


def object_name_for_entry(name: str) -> str:
    """Reproduce the button object name the previous sidebar exposed.

    The new toolbar-based sidebar keeps these ids (e.g. "Start experiment" ->
    "button_Start_experiment") so existing GUI tests and QSS that look buttons
    up by object name keep working during the migration.
    """
    return f"button_{name.replace(' ', '_')}"


class Sidebar(QToolBar):
    page_requested = Signal(str)
    external_plot_requested = Signal()
    status_entry_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._action_group = QActionGroup(self)
        self._action_group.setExclusive(True)
        self._actions: dict[str, QAction] = {}
        self._plot_button: QToolButton | None = None
        self._status_button: QToolButton | None = None
        self._status_entries: list[str] = []

        self._configure_toolbar()
        self._build_navigation_entries()
        self._enable_external_plot_trigger()
        self._capture_status_button()

    def _configure_toolbar(self) -> None:
        self.setObjectName("sidebar")
        self.setOrientation(Qt.Orientation.Vertical)
        self.setMovable(False)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)

    def _build_navigation_entries(self) -> None:
        for name, icon_file in NAVIGATION_ENTRIES:
            self._add_entry(name, load_icon(icon_file))

    def _enable_external_plot_trigger(self) -> None:
        plot_button = self.widgetForAction(self._actions[CREATE_PLOT])
        if isinstance(plot_button, QToolButton):
            plot_button.setToolTip("Right click to open external window")
            self._plot_button = plot_button
            plot_button.installEventFilter(self)

    def _capture_status_button(self) -> None:
        button = self.widgetForAction(self._actions[EXPERIMENT_STATUS])
        if isinstance(button, QToolButton):
            self._status_button = button

    @override
    def eventFilter(self, obj: QObject | None, event: QEvent | None) -> bool:
        if (
            obj is self._plot_button
            and isinstance(event, QMouseEvent)
            and event.type() == QEvent.Type.MouseButtonPress
            and event.button() == Qt.MouseButton.RightButton
        ):
            self.external_plot_requested.emit()
        return super().eventFilter(obj, event)

    def _add_entry(self, name: str, icon: QIcon) -> None:
        action = QAction(icon, name, self)
        action.setCheckable(True)
        action.setToolTip(name)
        # n=name binds the current name into each lambda instead of the loop variable
        action.triggered.connect(lambda _=False, n=name: self.page_requested.emit(n))
        self._action_group.addAction(action)
        self.addAction(action)
        self._actions[name] = action
        self._apply_button_object_name(action, name)

    def _apply_button_object_name(self, action: QAction, name: str) -> None:
        button = self.widgetForAction(action)
        if isinstance(button, QToolButton):
            button.setObjectName(object_name_for_entry(name))

    def action_for(self, name: str) -> QAction:
        return self._actions[name]

    def set_current(self, name: str) -> None:
        self._actions[name].setChecked(True)

    def set_status_enabled(self, enabled: bool) -> None:
        self._actions[EXPERIMENT_STATUS].setEnabled(enabled)

    def button_for(self, name: str) -> QToolButton | None:
        button = self.widgetForAction(self._actions[name])
        return button if isinstance(button, QToolButton) else None

    def add_status_entry(self, name: str) -> None:
        """Register a finished run under the experiment-status button.

        The first run keeps the status button as a plain page selector. From the
        second run onwards the button turns into a drop-down menu listing every
        run, with the most recent one marked bold.
        """
        self._status_entries.append(name)
        entry_count = len(self._status_entries)
        if entry_count == 2:
            self._create_status_menu()
            for entry in self._status_entries:
                self._add_status_menu_item(entry)
        elif entry_count > 2:
            self._add_status_menu_item(name)

    def _create_status_menu(self) -> None:
        if self._status_button is None:
            return
        menu = QMenu(self._status_button)
        self._status_button.setMenu(menu)
        self._status_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu.aboutToHide.connect(self._on_status_menu_about_to_hide)

    def _add_status_menu_item(self, name: str) -> None:
        if self._status_button is None:
            return
        menu = self._status_button.menu()
        if menu is None:
            return
        action_list = menu.actions()
        act = QAction(text=name, parent=menu)
        act.triggered.connect(
            lambda _=False, n=name: self.status_entry_selected.emit(n)
        )
        act.triggered.connect(lambda _: self._mark_action_bold(menu, act))
        if action_list:
            menu.insertAction(action_list[0], act)
        else:
            menu.addAction(act)
        self._mark_action_bold(menu, menu.actions()[0])

    def _on_status_menu_about_to_hide(self) -> None:
        if self._status_button is not None:
            QCoreApplication.sendEvent(self._status_button, QEvent(QEvent.Type.Leave))

    @staticmethod
    def _mark_action_bold(menu: QMenu, action_to_mark: QAction) -> None:
        for action in menu.actions():
            font = action.font()
            font.setBold(action is action_to_mark)
            action.setFont(font)
