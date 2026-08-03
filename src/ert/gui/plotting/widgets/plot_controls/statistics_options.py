from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from ert.gui.plotting.utils.qt_creator import (
    create_checkbox_with_tooltip,
    create_group_layout,
    create_labeled_row,
    create_spinbox_with_tooltip,
)
from ert.gui.plotting.utils.statistics_style import (
    DEFAULT_ENABLED_STATISTICS,
    STATISTICS,
)
from ert.gui.plotting.widgets.collapsible_section import CollapsibleSection

if TYPE_CHECKING:
    from ert.gui.plotting.utils import PlotConfig

logger = logging.getLogger(__name__)


class StatisticsOptions:
    """Owns the statistics selection, which persists across keys."""

    def __init__(self, connection_point: Callable[..., object]) -> None:
        self._toggles = {
            statistic: create_checkbox_with_tooltip(
                style.label,
                f"Show or hide the {style.label} "
                + ("band" if style.is_band else "line"),
                connection_point,
                initial_checked=statistic in DEFAULT_ENABLED_STATISTICS,
                logger=logger,
            )
            for statistic, style in STATISTICS.items()
        }
        self._area_toggle = create_checkbox_with_tooltip(
            "Area",
            "Draw the standard deviation, min/max and percentile ranges as a "
            "filled area instead of a pair of lines",
            connection_point,
            initial_checked=False,
            logger=logger,
        )
        self._std_dev_factor = create_spinbox_with_tooltip(
            "Std dev multiplier",
            "Choose the number of standard deviations to plot",
            connection_point,
            minimum=1,
            maximum=3,
            initial_value=1,
            logger=logger,
        )

        self._statistics_options = CollapsibleSection(
            "Statistics options",
            create_group_layout(
                [
                    *self._toggles.values(),
                    self._area_toggle,
                    create_labeled_row("Std dev multiplier", self._std_dev_factor),
                ]
            ),
        )

    def apply_to(self, plot_config: PlotConfig) -> None:
        plot_config.set_standard_deviation_factor(self._std_dev_factor.value())
        plot_config.set_statistics_options(
            self._enabled_statistics(),
            fill_bands=self._area_toggle.isChecked(),
        )

    def _enabled_statistics(self) -> set[str]:
        return {
            statistic
            for statistic, checkbox in self._toggles.items()
            if checkbox.isChecked()
        }

    def get_widget(self) -> CollapsibleSection:
        return self._statistics_options
