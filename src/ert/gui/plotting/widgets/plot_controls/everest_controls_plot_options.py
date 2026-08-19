import logging
from collections.abc import Callable

from PyQt6.QtWidgets import (
    QButtonGroup,
    QRadioButton,
)

from ert.gui.plotting.utils import PlotContext
from ert.gui.plotting.utils.qt_creator import (
    create_group_layout,
)
from ert.gui.plotting.widgets.collapsible_section import CollapsibleSection
from ert.gui.utils import log_once

logger = logging.getLogger(__name__)


class EverestControlsPlotOptions:
    def __init__(self, connection_point: Callable[..., object]) -> None:

        self._display_over_batches_radio = QRadioButton("batches")
        self._display_over_batches_radio.setObjectName("display_over_batches_radio")
        self._display_over_batches_radio.setChecked(True)
        self._display_over_controls_radio = QRadioButton("controls")
        self._display_over_controls_radio.setObjectName("display_over_controls_radio")
        self._display_over_button_group = QButtonGroup()
        self._display_over_button_group.addButton(self._display_over_batches_radio)
        self._display_over_button_group.addButton(self._display_over_controls_radio)
        self._display_over_button_group.buttonClicked.connect(connection_point)
        log_once(
            self._display_over_button_group.buttonClicked,
            logger,
            "Plot sidebar option used: 'X-axis display option'",
        )
        self._display_over_group = CollapsibleSection(
            "X-axis:",
            create_group_layout(
                [
                    self._display_over_batches_radio,
                    self._display_over_controls_radio,
                ]
            ),
        )

    def get_widget(self) -> CollapsibleSection:
        return self._display_over_group

    def update_plot_context(self, plot_context: PlotContext) -> None:
        plot_context.by_batch = self.is_batches_selected()

    def is_batches_selected(self) -> bool:
        return self._display_over_batches_radio.isChecked()
