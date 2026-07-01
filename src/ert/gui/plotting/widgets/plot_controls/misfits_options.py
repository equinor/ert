from collections.abc import Callable

from PyQt6.QtWidgets import (
    QCheckBox,
    QGroupBox,
)

from ert.gui.plotting.utils.qt_creator import create_group_box, create_group_layout


class MisfitsOptions:
    def __init__(self, connection_point: Callable[..., object]) -> None:

        self._toggle_mean = QCheckBox("Show mean")
        self._toggle_mean.setChecked(True)
        self._toggle_mean.stateChanged.connect(connection_point)
        self._toggle_outliers = QCheckBox("Show outliers")
        self._toggle_outliers.setChecked(True)
        self._toggle_outliers.stateChanged.connect(connection_point)
        self._toggle_scatter_plot = QCheckBox("Show scatter")
        self._toggle_scatter_plot.setChecked(False)
        self._toggle_scatter_plot.stateChanged.connect(connection_point)
        self._toggle_box = QCheckBox("Show box plot")
        self._toggle_box.setChecked(True)
        self._toggle_box.stateChanged.connect(connection_point)

        self._misfit_options = create_group_box(
            "Misfit options",
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
        return self._misfit_options
