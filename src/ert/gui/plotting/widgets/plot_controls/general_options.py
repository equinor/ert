import logging
from collections.abc import Callable

from PyQt6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QWidget,
)

from ert.gui.plotting.utils.qt_creator import create_group_box, create_group_layout

logger = logging.getLogger(__name__)


class GeneralOptions:
    def __init__(
        self, connection_point: Callable[..., object], *, is_everest: bool
    ) -> None:
        self._toggle_legend = QCheckBox("Legend")
        self._toggle_legend.setObjectName("legend_checkbox")
        self._toggle_legend.setChecked(True)
        self._toggle_legend.stateChanged.connect(connection_point)

        self._toggle_grid = QCheckBox("Grid")
        self._toggle_grid.setObjectName("grid_checkbox")
        self._toggle_grid.setChecked(True)
        self._toggle_grid.stateChanged.connect(connection_point)

        self._toggle_history = QCheckBox("History")
        self._toggle_history.setObjectName("history_checkbox")
        self._toggle_history.setChecked(True)
        self._toggle_history.stateChanged.connect(connection_point)

        self._toggle_observations = QCheckBox("Observations")
        self._toggle_observations.setObjectName("observations_checkbox")
        self._toggle_observations.setChecked(True)
        self._toggle_observations.stateChanged.connect(connection_point)

        self._toggle_log_scale = QCheckBox("Log scale")
        self._toggle_log_scale.setObjectName("log_scale_checkbox")
        self._toggle_log_scale.setChecked(False)
        self._toggle_log_scale.setVisible(False)

        self._toggle_log_scale.setToolTip("Toggle data domain to log scale and back")

        def log_log_scale_usage(_checked: bool) -> None:
            logger.info("Plot sidebar option used: 'Log scale'")
            self._toggle_log_scale.clicked.disconnect(log_log_scale_usage)

        self._toggle_log_scale.clicked.connect(log_log_scale_usage)
        self._toggle_log_scale.stateChanged.connect(connection_point)

        widgets: list[QWidget] = [
            self._toggle_legend,
            self._toggle_grid,
            self._toggle_log_scale,
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

    @property
    def log_checkbox_state(self) -> bool:
        return self._toggle_log_scale.isChecked()

    def set_log_visible(self, visible: bool) -> None:
        self._toggle_log_scale.setVisible(visible)

    def set_history_visible(self, visible: bool) -> None:
        self._toggle_history.setVisible(visible)

    def set_observations_visible(self, visible: bool) -> None:
        self._toggle_observations.setVisible(visible)

    def is_log_visible(self) -> bool:
        return self._toggle_log_scale.isVisible()
