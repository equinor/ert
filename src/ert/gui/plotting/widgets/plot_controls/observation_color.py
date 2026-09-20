import logging
from collections.abc import Callable

from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QWidget

from ert.gui.plotting.utils.plot_config import PlotConfig
from ert.gui.plotting.widgets.plot_controls.color_chooser import ColorBox
from ert.gui.utils import log_once

logger = logging.getLogger(__name__)


class ObservationColorEdit(QWidget):
    def __init__(
        self,
        connection_point: Callable[..., object],
        observation_checkbox: QCheckBox,
    ) -> None:
        super().__init__()

        self._observations_color_box = ColorBox(size=15)
        self._observations_color_box.color = PlotConfig().observations_color()
        self._observations_color_box.colorChanged.connect(
            lambda _color: connection_point()
        )
        log_once(
            self._observations_color_box.colorChanged,
            logger,
            "Plot sidebar option used: 'Observation color'",
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 0, 0)
        layout.setSpacing(1)

        color_row = QHBoxLayout()
        color_row.setContentsMargins(0, 0, 0, 0)
        color_row.setSpacing(6)
        color_label = QLabel("Change observation color:")
        label_font = color_label.font()
        label_font.setItalic(True)
        label_font.setPointSizeF(label_font.pointSizeF() * 0.9)
        color_label.setFont(label_font)
        color_row.addWidget(color_label)
        color_row.addWidget(self._observations_color_box)
        color_row.addStretch()

        layout.addLayout(color_row)

        observation_checkbox.toggled.connect(self.setVisible)
        self.setVisible(observation_checkbox.isChecked())

    def get_observations_color(self) -> tuple[str, float]:
        return (
            self._observations_color_box.color.name(),
            self._observations_color_box.color.alphaF(),
        )
