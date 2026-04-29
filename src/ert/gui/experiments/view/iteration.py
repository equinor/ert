from __future__ import annotations

from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from _ert.hook_runtime import HookRuntime

from .realization import RealizationWidget
from .update import UpdateWidget
from .workflow import WorkflowWidget, workflow_tab_title


class IterationWidget(QWidget):
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

    def ensure_realization_widget(self) -> RealizationWidget:
        for index in range(self._tab_widget.count()):
            widget = self._tab_widget.widget(index)
            if isinstance(widget, RealizationWidget):
                return widget

        widget = RealizationWidget(self.iteration, self)
        self._insert_tab(widget, "Run", self._tab_order(widget))
        return widget

    def ensure_update_widget(self) -> UpdateWidget:
        for index in range(self._tab_widget.count()):
            widget = self._tab_widget.widget(index)
            if isinstance(widget, UpdateWidget):
                return widget

        widget = UpdateWidget(self.iteration, self)
        self._insert_tab(widget, "Update", self._tab_order(widget))
        return widget

    def ensure_workflow_widget(
        self,
        hook: HookRuntime,
        workflow_names: list[str] | None = None,
    ) -> WorkflowWidget:
        for index in range(self._tab_widget.count()):
            widget = self._tab_widget.widget(index)
            if isinstance(widget, WorkflowWidget) and widget.hook == hook:
                return widget

        widget = WorkflowWidget(
            hook,
            iteration=self.iteration,
            workflow_names=workflow_names,
            parent=self,
        )
        self._insert_tab(widget, workflow_tab_title(hook), self._tab_order(widget))
        return widget

    def _insert_tab(self, widget: QWidget, title: str, order: int) -> None:
        insert_at = self._tab_widget.count()
        for index in range(self._tab_widget.count()):
            existing_widget = self._tab_widget.widget(index)
            if existing_widget is not None and self._tab_order(existing_widget) > order:
                insert_at = index
                break

        self._tab_widget.insertTab(insert_at, widget, title)
        self._tab_widget.setCurrentWidget(widget)

    def _tab_order(self, widget: QWidget) -> int:
        if isinstance(widget, WorkflowWidget):
            return {
                HookRuntime.PRE_SIMULATION: 0,
                HookRuntime.POST_SIMULATION: 20,
                HookRuntime.PRE_FIRST_UPDATE: 30,
                HookRuntime.PRE_UPDATE: 30,
                HookRuntime.POST_UPDATE: 50,
            }.get(widget.hook, 60)

        if isinstance(widget, RealizationWidget):
            return 10

        if isinstance(widget, UpdateWidget):
            return 40

        return 100
