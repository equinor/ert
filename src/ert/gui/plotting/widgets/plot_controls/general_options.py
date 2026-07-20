import logging
from collections.abc import Callable

from PyQt6.QtWidgets import QCheckBox, QGroupBox, QLabel, QVBoxLayout, QWidget

from ert.gui.plotting.utils.qt_creator import create_group_box, create_group_layout

from .plot_color_palette_selector import PlotColorPaletteSelector

logger = logging.getLogger(__name__)


class GeneralPlotOptions:
    def __init__(
        self, connection_point: Callable[..., object], *, is_everest: bool
    ) -> None:
        def log_option_usage(checkbox: QCheckBox, option_name: str) -> None:
            def log_usage(_checked: bool) -> None:
                logger.info("Plot sidebar option used: '%s'", option_name)
                checkbox.clicked.disconnect(log_usage)

            checkbox.clicked.connect(log_usage)

        self._toggle_legend = QCheckBox("Show Legend")
        self._toggle_legend.setObjectName("legend_checkbox")
        self._toggle_legend.setChecked(True)
        self._toggle_legend.setToolTip("Show or hide the legend")
        log_option_usage(self._toggle_legend, "Legend")
        self._toggle_legend.stateChanged.connect(connection_point)

        self._toggle_grid = QCheckBox("Show Grid")
        self._toggle_grid.setObjectName("grid_checkbox")
        self._toggle_grid.setChecked(True)
        self._toggle_grid.setToolTip("Show or hide the grid")
        log_option_usage(self._toggle_grid, "Grid")
        self._toggle_grid.stateChanged.connect(connection_point)

        self._toggle_history = QCheckBox("Show History")
        self._toggle_history.setObjectName("history_checkbox")
        self._toggle_history.setChecked(True)
        self._toggle_history.setToolTip("Show or hide history data")
        log_option_usage(self._toggle_history, "History")
        self._toggle_history.stateChanged.connect(connection_point)

        self._toggle_observations = QCheckBox("Show Observations")
        self._toggle_observations.setObjectName("observations_checkbox")
        self._toggle_observations.setChecked(True)
        self._toggle_observations.setToolTip("Show or hide observations")
        log_option_usage(self._toggle_observations, "Observations")
        self._toggle_observations.stateChanged.connect(connection_point)

        self._toggle_log_scale = QCheckBox("Log scale")
        self._toggle_log_scale.setObjectName("log_scale_checkbox")
        self._toggle_log_scale.setChecked(False)
        self._toggle_log_scale.setVisible(False)

        self._toggle_log_scale.setToolTip("Toggle data domain to log scale and back")
        log_option_usage(self._toggle_log_scale, "Log scale")
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

        palette_container = QWidget()
        palette_layout = QVBoxLayout(palette_container)
        palette_layout.setContentsMargins(0, 0, 0, 0)
        palette_layout.setSpacing(2)
        palette_layout.addWidget(QLabel("Selected color palette:"))
        self._color_cycle_selector = PlotColorPaletteSelector(connection_point)
        palette_layout.addWidget(self._color_cycle_selector)
        palette_layout.addWidget(self._color_cycle_selector.get_custom_palette_button())
        widgets.append(palette_container)

        self._general_options = create_group_box(
            "General options",
            create_group_layout(widgets),
        )
        self._general_options.setObjectName("general_options")

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

    def get_color_cycle(self) -> list[tuple[str, float]]:
        return self._color_cycle_selector.get_color_cycle()
