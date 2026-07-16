from collections.abc import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox

from ert.gui.plotting.utils.plot_color_palettes import PALETTES_WITH_DESCRIPTIONS


class PlotColorPaletteSelector(QComboBox):
    def __init__(self, connection_point: Callable[..., object]) -> None:
        super().__init__()

        self.setObjectName("plot_color_palette_selector")

        self.setToolTip(
            "Select a color palette for the plot lines. "
            "The selected palette will be applied to all plots."
        )

        for palette in PALETTES_WITH_DESCRIPTIONS:
            self.addItem(palette)
            self.setItemData(
                self.count() - 1,
                PALETTES_WITH_DESCRIPTIONS[palette][1],
                Qt.ItemDataRole.ToolTipRole,
            )
        self.activated.connect(connection_point)

    def get_color_cycle(self) -> list[tuple[str, float]]:
        return PALETTES_WITH_DESCRIPTIONS[self.currentText()][0]
