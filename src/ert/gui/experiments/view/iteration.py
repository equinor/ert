from __future__ import annotations

from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtWidgets import QComboBox, QStackedWidget, QTabWidget, QVBoxLayout, QWidget

from _ert.hook_runtime import HookRuntime

from .realization import RealizationWidget
from .update import UpdateWidget
from .workflow import WorkflowWidget, workflow_tab_title


class IterationWidget(QWidget):
    currentTabChanged = Signal()

    def __init__(self, iteration: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._iteration = iteration
        self._tab_widget = QTabWidget(self)
        self._tab_widget.currentChanged.connect(
            lambda _index: self.currentTabChanged.emit()
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tab_widget)

    @property
    def iteration(self) -> int:
        return self._iteration

    def current_widget(self) -> QWidget | None:
        return self._tab_widget.currentWidget()

    def ensure_realization_widget(self) -> RealizationWidget:
        for index in range(self._tab_widget.count()):
            widget = self._tab_widget.widget(index)
            if isinstance(widget, RealizationWidget):
                return widget

        widget = RealizationWidget(self._iteration, self)
        self._insert_tab(widget, "Run", self._tab_order(widget))
        return widget

    def ensure_update_widget(self) -> UpdateWidget:
        for index in range(self._tab_widget.count()):
            widget = self._tab_widget.widget(index)
            if isinstance(widget, UpdateWidget):
                return widget

        widget = UpdateWidget(self._iteration, self)
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
            iteration=self._iteration,
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


class IterationsWidget(QWidget):
    currentPageChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._selector = QComboBox(self)
        self._selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._stack = QStackedWidget(self)
        self._pages: dict[tuple[int, bool], IterationWidget] = {}

        self._selector.currentIndexChanged.connect(self._stack.setCurrentIndex)
        self._selector.currentIndexChanged.connect(
            lambda _index: self.currentPageChanged.emit()
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._selector)
        layout.addWidget(self._stack)

    def current_page(self) -> IterationWidget | None:
        widget = self._stack.currentWidget()
        return widget if isinstance(widget, IterationWidget) else None

    def current_widget(self) -> QWidget | None:
        page = self.current_page()
        return page.current_widget() if page is not None else None

    def ensure_run_page(self, iteration: int) -> IterationWidget:
        return self._ensure_page(iteration, is_update=False)

    def ensure_update_page(self, iteration: int) -> IterationWidget:
        return self._ensure_page(iteration, is_update=True)

    def set_current_page(self, page: IterationWidget) -> None:
        index = self._stack.indexOf(page)
        if index >= 0:
            self._selector.setCurrentIndex(index)

    def _ensure_page(self, iteration: int, is_update: bool) -> IterationWidget:
        key = (iteration, is_update)
        existing = self._pages.get(key)
        if existing is not None:
            return existing

        page = IterationWidget(iteration, self)
        page.currentTabChanged.connect(self.currentPageChanged)

        insert_at = self._stack.count()
        for index in range(self._stack.count()):
            existing_widget = self._stack.widget(index)
            if not isinstance(existing_widget, IterationWidget):
                continue

            existing_key = self._page_key(existing_widget)
            if existing_key is not None and existing_key > key:
                insert_at = index
                break

        self._pages[key] = page
        self._stack.insertWidget(insert_at, page)
        self._selector.insertItem(insert_at, self._page_title(iteration, is_update))
        self.set_current_page(page)
        return page

    def _page_key(self, page: IterationWidget) -> tuple[int, bool] | None:
        for key, widget in self._pages.items():
            if widget is page:
                return key
        return None

    @staticmethod
    def _page_title(iteration: int, is_update: bool) -> str:
        return f"update-{iteration}" if is_update else f"iteration-{iteration}"
