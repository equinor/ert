from PyQt6.QtWidgets import (
    QHBoxLayout,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ert.gui.experiments.view.iteration_selector import IterationSelector
from ert.gui.experiments.view.update import UpdateWidget


class UpdateWidgetStack(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._stack = QStackedWidget(self)

        self._iteration_selector = IterationSelector(self)
        self._iteration_selector.currentIndexChanged.connect(
            self._stack.setCurrentIndex
        )

        selector_layout = QHBoxLayout()
        selector_layout.addWidget(self._iteration_selector)
        selector_layout.addStretch()

        layout = QVBoxLayout()
        layout.addLayout(selector_layout)
        layout.addWidget(self._stack)
        self.setLayout(layout)

    def add_update_widget(self, iteration: int) -> UpdateWidget:
        widget = UpdateWidget(iteration)
        self._stack.addWidget(widget)
        self._iteration_selector.add_iteration(f"Update {iteration}", iteration)
        return widget

    def get_update_widget_for_iteration(self, iteration: int) -> UpdateWidget:
        for i in range(self._stack.count()):
            widget = self._stack.widget(i)
            if isinstance(widget, UpdateWidget) and widget.iteration == iteration:
                return widget
        raise ValueError("Could not find UpdateWidget")
