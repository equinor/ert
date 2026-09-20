from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QCheckBox,
)

if TYPE_CHECKING:
    from ert.gui.plotting.utils.plot_context import PlotContext

from ert.gui.plotting.utils.qt_creator import create_group_layout
from ert.gui.plotting.widgets.collapsible_section import CollapsibleSection

logger = logging.getLogger(__name__)

NAME_AND_TOOLTIP = [
    (
        "histogram_checkbox",
        "Show histogram",
        "Adds a histogram of the data to the plot.\nDisplayed as counts.",
    ),
    (
        "gkde_checkbox",
        "Show estimated density",
        (
            "Adds a Gaussian kernel density estimate to the plot."
            "\nDisplays a line for the probability density function"
            " of the data for each ensemble."
        ),
    ),
    (
        "rug_checkbox",
        "Show individual points",
        (
            "Displays the distribution as a set of individual points for each ensemble."
            "\nIf histogram/Gaussian KDE is enabled, "
            "the data points will be plotted on its own axis below the main plot."
        ),
    ),
]


class DistributionOptions:
    def __init__(self, connection_point: Callable[..., object]) -> None:
        self._logged_options: set[str] = set()

        self._histogram, self._gkde, self._rug_plot = [
            self._add_checkbox(obj_name, label, tooltip, connection_point)
            for obj_name, label, tooltip in NAME_AND_TOOLTIP
        ]
        self._distribution_options = CollapsibleSection(
            "Distribution options",
            create_group_layout(
                [
                    self._histogram,
                    self._gkde,
                    self._rug_plot,
                ]
            ),
            expanded=True,
        )

    @property
    def histogram_checkbox_state(self) -> bool:
        return self._histogram.isChecked()

    @histogram_checkbox_state.setter
    def histogram_checkbox_state(self, value: bool) -> None:
        self._histogram.setChecked(value)

    @property
    def gkde_checkbox_state(self) -> bool:
        return self._gkde.isChecked()

    @gkde_checkbox_state.setter
    def gkde_checkbox_state(self, value: bool) -> None:
        self._gkde.setChecked(value)

    @property
    def rug_checkbox_state(self) -> bool:
        return self._rug_plot.isChecked()

    @rug_checkbox_state.setter
    def rug_checkbox_state(self, value: bool) -> None:
        self._rug_plot.setChecked(value)

    def get_widget(self) -> CollapsibleSection:
        return self._distribution_options

    # Only wish to log the first time a distribution option is used in a session,
    # otherwise could risk flooding the log
    def _log_usage(self, distribution_option_name: str, _checked: bool) -> None:
        if distribution_option_name not in self._logged_options:
            logger.info("Plot sidebar option used: '%s'", distribution_option_name)
            self._logged_options.add(distribution_option_name)

    def update_plot_context(self, plot_context: PlotContext) -> None:
        plot_context.histogram = self.histogram_checkbox_state
        plot_context.gkde_plot = self.gkde_checkbox_state
        plot_context.rug_plot = self.rug_checkbox_state

    def _add_checkbox(
        self,
        obj_name: str,
        label: str,
        tooltip: str,
        connection_point: Callable[..., object],
    ) -> QCheckBox:
        checkbox = QCheckBox(f"{label}")
        checkbox.setObjectName(f"{obj_name}")

        checkbox.setToolTip(tooltip)
        checkbox.setChecked(True)

        checkbox.stateChanged.connect(connection_point)
        checkbox.clicked.connect(
            lambda checked: self._log_usage(f"Distribution option: {label}", checked)
        )
        return checkbox
