import logging
from collections.abc import Callable

from PyQt6.QtWidgets import (
    QGroupBox,
)

from ert.gui.plotting.utils.qt_creator import (
    create_checkbox_with_tooltip,
    create_group_box,
    create_group_layout,
)

logger = logging.getLogger(__name__)


class BoxplotOptions:
    def __init__(self, connection_point: Callable[..., object]) -> None:

        (
            self._toggle_mean,
            self._toggle_outliers,
            self._toggle_scatter_plot,
            self._toggle_box,
        ) = [
            create_checkbox_with_tooltip(
                name, tooltip, connection_point, initial_checked=checked, logger=logger
            )
            for name, tooltip, checked in [
                ("Mean", "Show or hide the mean", True),
                ("Outliers", "Show or hide outliers", True),
                ("Scatter points", "Show or hide scatter points", False),
                ("Boxplot", "Show or hide the boxplot", True),
            ]
        ]

        self._boxplot_options = create_group_box(
            "Boxplot options",
            create_group_layout(
                [
                    self._toggle_scatter_plot,
                    self._toggle_box,
                    self._toggle_mean,
                    self._toggle_outliers,
                ]
            ),
        )

    @property
    def mean_checkbox_state(self) -> bool:
        return self._toggle_mean.isChecked()

    @mean_checkbox_state.setter
    def mean_checkbox_state(self, value: bool) -> None:
        self._toggle_mean.setChecked(value)

    @property
    def outliers_checkbox_state(self) -> bool:
        return self._toggle_outliers.isChecked()

    @outliers_checkbox_state.setter
    def outliers_checkbox_state(self, value: bool) -> None:
        self._toggle_outliers.setChecked(value)

    @property
    def scatter_checkbox_state(self) -> bool:
        return self._toggle_scatter_plot.isChecked()

    @scatter_checkbox_state.setter
    def scatter_checkbox_state(self, value: bool) -> None:
        self._toggle_scatter_plot.setChecked(value)

    @property
    def box_checkbox_state(self) -> bool:
        return self._toggle_box.isChecked()

    @box_checkbox_state.setter
    def box_checkbox_state(self, value: bool) -> None:
        self._toggle_box.setChecked(value)

    def get_widget(self) -> QGroupBox:
        return self._boxplot_options
