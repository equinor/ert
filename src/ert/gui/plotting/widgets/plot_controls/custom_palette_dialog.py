from collections.abc import Sequence

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ert.gui.plotting.customization_dialog.color_chooser import ColorBox
from ert.gui.plotting.utils.plot_color_palettes import MINIMUM_COLOR_CYCLE_LENGTH


class CustomPaletteDialog(QDialog):
    def __init__(
        self,
        color_cycle: Sequence[tuple[str, float]],
        parent: QWidget | None = None,
        *,
        number_of_colors: int = MINIMUM_COLOR_CYCLE_LENGTH,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit custom palette")
        self.setObjectName("custom_palette_dialog")

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select colors for the custom palette:"))

        color_layout = QHBoxLayout()
        self._color_boxes: list[ColorBox] = []
        for i in range(number_of_colors):
            color_box = ColorBox(size=20)
            color_box.setToolTip(f"#{i + 1}")
            if i < len(color_cycle):
                color_box.color = color_cycle[i]
            self._color_boxes.append(color_box)
            color_layout.addWidget(color_box)
        layout.addLayout(color_layout)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_color_cycle(self) -> list[tuple[str, float]]:
        return [(box.color.name(), box.color.alphaF()) for box in self._color_boxes]
