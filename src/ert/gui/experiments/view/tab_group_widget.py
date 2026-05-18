from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from _ert.hook_runtime import HookRuntime

from .realization import RealizationWidget
from .update import UpdateWidget
from .workflow import WorkflowWidget, workflow_tab_title

TabWidgetType = TypeVar(
    "TabWidgetType",
    RealizationWidget,
    UpdateWidget,
    WorkflowWidget,
)


class TabGroupWidget(QWidget):
    currentTabChanged = Signal()

    def __init__(self, iteration: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.iteration = iteration
        self._tab_widget = QTabWidget(self)
        self._tab_widget.currentChanged.connect(
            lambda _index: self.currentTabChanged.emit()
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tab_widget)

    def current_widget(self) -> QWidget | None:
        return self._tab_widget.currentWidget()

    def _select_or_create_tab_of_type(
        self,
        widget_type: type[TabWidgetType],
        title: str,
        create_widget: Callable[[], TabWidgetType],
        matches: Callable[[TabWidgetType], bool] | None = None,
    ) -> TabWidgetType:
        for index in range(self._tab_widget.count()):
            widget = self._tab_widget.widget(index)
            if isinstance(widget, widget_type) and (matches is None or matches(widget)):
                self._tab_widget.setCurrentWidget(widget)
                return widget

        widget = create_widget()
        self._tab_widget.addTab(widget, title)
        self._tab_widget.setCurrentWidget(widget)
        return widget

    def select_or_create_realization_tab(self) -> RealizationWidget:
        return self._select_or_create_tab_of_type(
            RealizationWidget,
            "Run",
            lambda: RealizationWidget(self.iteration, self),
        )

    def select_or_create_update_tab(self) -> UpdateWidget:
        return self._select_or_create_tab_of_type(
            UpdateWidget,
            "Update",
            lambda: UpdateWidget(self.iteration, self),
        )

    def select_or_create_workflow_tab(
        self,
        hook: HookRuntime,
    ) -> WorkflowWidget:
        return self._select_or_create_tab_of_type(
            WorkflowWidget,
            workflow_tab_title(hook),
            lambda: WorkflowWidget(hook, parent=self),
            matches=lambda widget: widget.hook == hook,
        )
