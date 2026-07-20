from collections.abc import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QPushButton

from ert.gui.icon_utils import load_icon
from ert.gui.plotting.utils.plot_color_palettes import PALETTES_WITH_DESCRIPTIONS
from ert.gui.plotting.widgets.plot_controls import CustomPaletteDialog


class PlotColorPaletteSelector(QComboBox):
    CUSTOM_PALETTE_NAME = "Custom"

    def __init__(self, connection_point: Callable[..., object]) -> None:
        super().__init__()

        self._connection_point = connection_point

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
        self.custom_palette_button = QPushButton(
            load_icon("add_circle_outlined.svg"), "Create custom palette"
        )
        self.custom_palette_button.setToolTip(
            "Edit the custom color palette. "
            "The selected palette will be applied to all plots."
        )
        self.custom_palette_button.clicked.connect(self._open_custom_palette_dialog)

    def _open_custom_palette_dialog(self) -> None:
        if self.CUSTOM_PALETTE_NAME in PALETTES_WITH_DESCRIPTIONS:
            dialog = CustomPaletteDialog(
                PALETTES_WITH_DESCRIPTIONS[self.CUSTOM_PALETTE_NAME][0], self
            )
        else:
            dialog = CustomPaletteDialog(self.get_color_cycle(), self)
        if dialog.exec():
            self.edit_palette(dialog.get_color_cycle())
            self.setCurrentText(self.CUSTOM_PALETTE_NAME)
            self._connection_point()  # Needed to trigger redraw

    def get_color_cycle(self) -> list[tuple[str, float]]:
        return PALETTES_WITH_DESCRIPTIONS[self.currentText()][0]

    def edit_palette(self, color_cycle: list[tuple[str, float]]) -> None:
        PALETTES_WITH_DESCRIPTIONS[self.CUSTOM_PALETTE_NAME] = (
            color_cycle,
            "Custom color palette. Applied to all plots.",
        )
        # Only want 1 custom palette
        if self.findText(self.CUSTOM_PALETTE_NAME, Qt.MatchFlag.MatchFixedString) == -1:
            self.addItem(self.CUSTOM_PALETTE_NAME)
            self.setItemData(
                self.count() - 1,
                PALETTES_WITH_DESCRIPTIONS[self.CUSTOM_PALETTE_NAME][1],
                Qt.ItemDataRole.ToolTipRole,
            )
        self.custom_palette_button.setIcon(load_icon("edit.svg"))
        self.custom_palette_button.setText("Edit custom palette")

    def get_custom_palette_button(self) -> "QPushButton":
        return self.custom_palette_button
