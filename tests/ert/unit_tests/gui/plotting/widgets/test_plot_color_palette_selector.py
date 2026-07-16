from unittest.mock import Mock

from PyQt6.QtCore import Qt

from ert.gui.plotting.utils.plot_color_palettes import PALETTES_WITH_DESCRIPTIONS
from ert.gui.plotting.widgets.plot_controls.plot_color_palette_selector import (
    PlotColorPaletteSelector,
)


def test_that_color_cycle_selector_defaults_to_the_default_palette(qtbot):
    selector = PlotColorPaletteSelector(Mock())
    qtbot.addWidget(selector)

    assert selector.currentText() == next(iter(PALETTES_WITH_DESCRIPTIONS.keys()))
    assert (
        selector.get_color_cycle()
        == PALETTES_WITH_DESCRIPTIONS[selector.currentText()][0]
    )


def test_that_palette_selector_color_cycle_matches_selected_palette(qtbot):
    selector = PlotColorPaletteSelector(Mock())
    qtbot.addWidget(selector)

    for palette in PALETTES_WITH_DESCRIPTIONS:
        selector.setCurrentText(palette)
        assert selector.get_color_cycle() == PALETTES_WITH_DESCRIPTIONS[palette][0]


def test_that_changing_selection_via_keyboard_invokes_the_connection_point(qtbot):
    connection_point = Mock()
    selector = PlotColorPaletteSelector(connection_point)
    qtbot.addWidget(selector)
    selector.show()

    qtbot.keyClick(selector, Qt.Key.Key_Down)

    connection_point.assert_called_once()


def test_that_each_palette_item_tooltip_matches_description_mapping(qtbot):
    selector = PlotColorPaletteSelector(Mock())
    qtbot.addWidget(selector)

    for index in range(selector.count()):
        tooltip = selector.itemData(index, Qt.ItemDataRole.ToolTipRole)
        assert tooltip == PALETTES_WITH_DESCRIPTIONS[selector.itemText(index)][1]
