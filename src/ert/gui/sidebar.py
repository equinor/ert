from __future__ import annotations

from typing import override

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtGui import QAction, QActionGroup, QIcon, QMouseEvent
from PyQt6.QtWidgets import QToolBar, QToolButton, QWidget

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

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._action_group = QActionGroup(self)
        self._action_group.setExclusive(True)
        self._actions: dict[str, QAction] = {}
        self._plot_button: QToolButton | None = None

        self._configure_toolbar()
        self._build_navigation_entries()
        self._enable_external_plot_trigger()

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
