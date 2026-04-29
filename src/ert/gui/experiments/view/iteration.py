from __future__ import annotations

from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from _ert.hook_runtime import HookRuntime

from .realization import RealizationWidget
from .update import UpdateWidget
from .workflow import WorkflowWidget


class IterationWidget(QWidget):
    currentTabChanged = Signal()

    def __init__(self, iteration: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.iteration = iteration
        self.is_update_page = False
        self._tab_widget = QTabWidget(self)
        self._tab_widget.currentChanged.connect(
            lambda _index: self.currentTabChanged.emit()
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tab_widget)

    def current_widget(self) -> QWidget | None:
        return self._tab_widget.currentWidget()

    def select_or_create_realization_tab(self) -> RealizationWidget:
        for index in range(self._tab_widget.count()):
            widget = self._tab_widget.widget(index)
            if isinstance(widget, RealizationWidget):
                self._tab_widget.setCurrentWidget(widget)
                return widget

        widget = RealizationWidget(self.iteration, self)
        self._tab_widget.addTab(widget, "Run")
        self._tab_widget.setCurrentWidget(widget)
        return widget

    def select_or_create_update_tab(self) -> UpdateWidget:
        for index in range(self._tab_widget.count()):
            widget = self._tab_widget.widget(index)
            if isinstance(widget, UpdateWidget):
                self._tab_widget.setCurrentWidget(widget)
                return widget

        widget = UpdateWidget(self.iteration, self)
        self._tab_widget.addTab(widget, "Update")
        self._tab_widget.setCurrentWidget(widget)
        return widget

    def select_or_create_workflow_tab(
        self,
        hook: HookRuntime,
        workflow_names: list[str] | None = None,
    ) -> WorkflowWidget:
        for index in range(self._tab_widget.count()):
            widget = self._tab_widget.widget(index)
            if isinstance(widget, WorkflowWidget) and widget.hook == hook:
                self._tab_widget.setCurrentWidget(widget)
                return widget

        widget = WorkflowWidget(
            hook,
            workflow_names=workflow_names,
            parent=self,
        )
        self._tab_widget.addTab(widget, hook.workflow_tab_title())
        self._tab_widget.setCurrentWidget(widget)
        return widget
