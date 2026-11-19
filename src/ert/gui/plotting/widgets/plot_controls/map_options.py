from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from ert.gui.plotting.utils.qt_creator import (
    create_checkbox_with_tooltip,
    create_group_layout,
)
from ert.gui.plotting.widgets.collapsible_section import CollapsibleSection

if TYPE_CHECKING:
    from ert.gui.plotting.utils import PlotContext

logger = logging.getLogger(__name__)


class MapOptions:
    def __init__(self, connection_point: Callable[..., object]) -> None:
        self._observations_map, self._misfit_map, self._mean_simulated_response_map = [
            create_checkbox_with_tooltip(
                name, tooltip, connection_point, initial_checked=True, logger=logger
            )
            for name, tooltip in [
                (
                    "Observations map",
                    "Show or hide the map of the observed values.",
                ),
                (
                    "Misfit map",
                    (
                        "Show or hide the map of the mean signed chi-squared misfit\n"
                        "between the responses and the observations."
                    ),
                ),
                (
                    "Mean simulated response map",
                    "Show or hide the map of the mean simulated responses.",
                ),
            ]
        ]

        self._map_options = CollapsibleSection(
            "Map options",
            create_group_layout(
                [
                    self._observations_map,
                    self._misfit_map,
                    self._mean_simulated_response_map,
                ]
            ),
            expanded=True,
        )

    @property
    def misfit_map_checkbox_state(self) -> bool:
        return self._misfit_map.isChecked()

    @property
    def observations_map_checkbox_state(self) -> bool:
        return self._observations_map.isChecked()

    @property
    def mean_simulated_response_map_checkbox_state(self) -> bool:
        return self._mean_simulated_response_map.isChecked()

    def update_plot_context(self, plot_context: PlotContext) -> None:
        plot_context.misfit_map = self.misfit_map_checkbox_state
        plot_context.observations_map = self.observations_map_checkbox_state
        plot_context.mean_simulated_response_map = (
            self.mean_simulated_response_map_checkbox_state
        )

    def get_widget(self) -> CollapsibleSection:
        return self._map_options
