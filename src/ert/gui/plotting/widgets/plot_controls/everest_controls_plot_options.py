from collections.abc import Callable

from PyQt6.QtWidgets import (
    QButtonGroup,
    QGroupBox,
    QRadioButton,
)

from ert.gui.plotting.utils.qt_creator import create_group_box, create_group_layout


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

        self._display_over_group = create_group_box(
            "X-axis:",
            create_group_layout(
                [
                    self._display_over_batches_radio,
                    self._display_over_controls_radio,
                ]
            ),
        )

    def get_widget(self) -> QGroupBox:
        return self._display_over_group

    def is_batches_selected(self) -> bool:
        return self._display_over_batches_radio.isChecked()
