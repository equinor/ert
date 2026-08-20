import logging
from collections.abc import Callable

from PyQt6.QtCore import QObject
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ert.gui.plotting.utils import PlotContext
from ert.gui.plotting.utils.qt_creator import (
    create_checkbox_with_tooltip,
    create_group_layout,
)
from ert.gui.plotting.widgets.collapsible_section import CollapsibleSection
from ert.gui.utils import is_everest_application

from .observation_color import ObservationColorEdit
from .plot_color_palette_selector import PlotColorPaletteSelector

logger = logging.getLogger(__name__)

DEFAULT_EXPANDED_STATE = True


class GeneralPlotOptions(QObject):
    axisLabelEditRequested = Signal(str)
    titleEditRequested = Signal()

    def __init__(
        self,
        connection_point: Callable[..., object],
    ) -> None:
        super().__init__()

        self._is_everest = is_everest_application()

        def create_edit_button(
            name: str,
            callback: Callable[[], None],
        ) -> QPushButton:
            button = QPushButton(f"Edit {name}")
            button.setObjectName(f"change_{name.replace('-', '_')}_button")
            button.clicked.connect(callback)
            return button

        (
            self._toggle_legend,
            self._toggle_grid,
            self._toggle_history,
            self._toggle_observations,
            self._toggle_log_scale,
        ) = [
            create_checkbox_with_tooltip(
                name, tooltip, connection_point, initial_checked=checked, logger=logger
            )
            for name, tooltip, checked in [
                ("Legend", "Show or hide the legend", True),
                ("Grid", "Show or hide the grid", True),
                ("History", "Show or hide history data", True),
                ("Observations", "Show or hide observations", True),
                ("Log scale", "Toggle data domain to log scale and back", False),
            ]
        ]

        edit_buttons = QWidget()
        edit_buttons_layout = QHBoxLayout(edit_buttons)
        edit_buttons_layout.setContentsMargins(0, 0, 0, 0)

        for name, callback in (
            ("x-label", lambda: self.axisLabelEditRequested.emit("x")),
            ("y-label", lambda: self.axisLabelEditRequested.emit("y")),
            ("title", self.titleEditRequested.emit),
        ):
            edit_buttons_layout.addWidget(create_edit_button(name, callback))

        widgets: list[QWidget] = [
            self._toggle_legend,
            self._toggle_grid,
            self._toggle_log_scale,
        ]

        if not self._is_everest:
            self._observations_color_edit = ObservationColorEdit(
                connection_point=connection_point,
                observation_checkbox=self._toggle_observations,
            )
            widgets.extend(
                [
                    self._toggle_history,
                    self._toggle_observations,
                    self._observations_color_edit,
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

        widgets.extend([palette_container, edit_buttons])

        self._general_options = CollapsibleSection(
            "General options",
            create_group_layout(widgets),
            expanded=DEFAULT_EXPANDED_STATE,
        )
        self._general_options.setObjectName("general_options")

    def get_widget(self) -> CollapsibleSection:
        return self._general_options

    def get_text_input(
        self,
        title: str,
        prompt: str,
        current_text: str | None,
    ) -> tuple[str, bool]:
        dialog = QInputDialog(self._general_options)
        dialog.setWindowTitle(title)
        dialog.setLabelText(prompt)
        dialog.setTextValue(current_text or "")
        size_hint = dialog.sizeHint()
        title_width = dialog.fontMetrics().horizontalAdvance(title)
        dialog.resize(
            max(size_hint.width(), title_width + 175),
            size_hint.height(),
        )
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        return dialog.textValue() or "", accepted

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

    def update_plot_context(
        self,
        plot_context: PlotContext,
        *,
        history_data_available: bool,
        has_observations: bool,
        show_observations: bool,
        log_scale_available: bool,
    ) -> None:
        plot_config = plot_context.plotConfig()
        plot_config.set_legend_enabled(self.legend_checkbox_state)
        plot_config.set_grid_enabled(self.grid_checkbox_state)
        plot_config.set_line_color_cycle(self.get_color_cycle())

        self._toggle_log_scale.setVisible(log_scale_available)
        plot_context.log_scale = self.log_checkbox_state and log_scale_available

        if self._is_everest:
            return

        self._toggle_history.setVisible(history_data_available)
        self._toggle_observations.setVisible(show_observations)
        self._observations_color_edit.setVisible(
            show_observations and self.observations_checkbox_state
        )
        plot_config.set_history_enabled(
            self.history_checkbox_state and history_data_available
        )
        plot_config.set_observations_enabled(
            self.observations_checkbox_state and has_observations
        )
        plot_config.set_observations_color(self.get_observations_color())

    def get_color_cycle(self) -> list[tuple[str, float]]:
        return self._color_cycle_selector.get_color_cycle()

    def get_observations_color(self) -> tuple[str, float]:
        return self._observations_color_edit.get_observations_color()
