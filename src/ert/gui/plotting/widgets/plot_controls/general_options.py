from collections.abc import Callable

from PyQt6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QWidget,
)

from ert.gui.plotting.utils.qt_creator import create_group_box, create_group_layout


class GeneralOptions:
    def __init__(
        self, connection_point: Callable[..., object], *, is_everest: bool
    ) -> None:
        def refresh_plot(_state: int) -> None:
            connection_point()

        self._toggle_legend = QCheckBox("Legend")
        self._toggle_legend.setChecked(True)
        self._toggle_legend.stateChanged.connect(refresh_plot)

        self._toggle_grid = QCheckBox("Grid")
        self._toggle_grid.setChecked(True)
        self._toggle_grid.stateChanged.connect(refresh_plot)

        self._toggle_history = QCheckBox("History")
        self._toggle_history.setChecked(True)
        self._toggle_history.stateChanged.connect(refresh_plot)

        self._toggle_observations = QCheckBox("Observations")
        self._toggle_observations.setChecked(True)
        self._toggle_observations.stateChanged.connect(refresh_plot)

        widgets: list[QWidget] = [
            self._toggle_legend,
            self._toggle_grid,
        ]
        if not is_everest:
            widgets.extend(
                [
                    self._toggle_history,
                    self._toggle_observations,
                ]
            )

        self._general_options = create_group_box(
            "General options",
            create_group_layout(widgets),
        )

    def get_widget(self) -> QGroupBox:
        return self._general_options

    @property
    def legend_checkbox_state(self) -> bool:
        return self._toggle_legend.isChecked()

    @property
    def grid_checkbox_state(self) -> bool:
        return self._toggle_grid.isChecked()

    @property
    def history_checkbox_state(self) -> bool:
        return self._toggle_history.isChecked()

    @property
    def observations_checkbox_state(self) -> bool:
        return self._toggle_observations.isChecked()
