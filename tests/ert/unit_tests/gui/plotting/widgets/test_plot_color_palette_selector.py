from unittest.mock import Mock

import pytest
from PyQt6.QtCore import Qt

from ert.gui.plotting.utils.plot_color_palettes import PALETTES_WITH_DESCRIPTIONS
from ert.gui.plotting.widgets.plot_controls import plot_color_palette_selector
from ert.gui.plotting.widgets.plot_controls.plot_color_palette_selector import (
    PlotColorPaletteSelector,
)


@pytest.fixture
def cleanup_custom_palette():
    yield
    PALETTES_WITH_DESCRIPTIONS.pop(PlotColorPaletteSelector.CUSTOM_PALETTE_NAME, None)


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


def test_that_editing_palette_registers_a_single_custom_entry(
    qtbot, cleanup_custom_palette
):
    selector = PlotColorPaletteSelector(Mock())
    qtbot.addWidget(selector)
    custom_name = PlotColorPaletteSelector.CUSTOM_PALETTE_NAME
    color_cycle = [("#111111", 1.0), ("#222222", 1.0)]

    selector.edit_palette(color_cycle)

    assert selector.findText(custom_name, Qt.MatchFlag.MatchFixedString) != -1
    assert PALETTES_WITH_DESCRIPTIONS[custom_name][0] == color_cycle


def test_that_re_editing_palette_does_not_duplicate_the_custom_entry(
    qtbot, cleanup_custom_palette
):
    selector = PlotColorPaletteSelector(Mock())
    qtbot.addWidget(selector)
    custom_name = PlotColorPaletteSelector.CUSTOM_PALETTE_NAME

    selector.edit_palette([("#111111", 1.0)])
    selector.edit_palette([("#333333", 1.0)])

    matches = [
        selector.itemText(i)
        for i in range(selector.count())
        if selector.itemText(i) == custom_name
    ]
    assert len(matches) == 1


def test_that_get_color_cycle_returns_edited_colors_when_custom_is_selected(
    qtbot, cleanup_custom_palette
):
    selector = PlotColorPaletteSelector(Mock())
    qtbot.addWidget(selector)
    custom_name = PlotColorPaletteSelector.CUSTOM_PALETTE_NAME
    color_cycle = [("#abcdef", 0.5), ("#123456", 1.0)]

    selector.edit_palette(color_cycle)
    selector.setCurrentText(custom_name)

    assert selector.get_color_cycle() == color_cycle


def test_that_accepting_the_palette_dialog_saves_selects_custom_and_redraws(
    qtbot, monkeypatch, cleanup_custom_palette
):
    connection_point = Mock()
    selector = PlotColorPaletteSelector(connection_point)
    qtbot.addWidget(selector)
    custom_name = PlotColorPaletteSelector.CUSTOM_PALETTE_NAME
    edited_cycle = [("#0a0a0a", 1.0), ("#0b0b0b", 1.0)]

    fake_dialog = Mock()
    fake_dialog.exec.return_value = True
    fake_dialog.get_color_cycle.return_value = edited_cycle
    monkeypatch.setattr(
        plot_color_palette_selector,
        "CustomPaletteDialog",
        Mock(return_value=fake_dialog),
    )

    selector._open_custom_palette_dialog()

    assert selector.currentText() == custom_name
    assert PALETTES_WITH_DESCRIPTIONS[custom_name][0] == edited_cycle
    connection_point.assert_called_once()


def test_that_cancelling_the_palette_dialog_does_not_register_a_custom_entry(
    qtbot, monkeypatch, cleanup_custom_palette
):
    connection_point = Mock()
    selector = PlotColorPaletteSelector(connection_point)
    qtbot.addWidget(selector)
    custom_name = PlotColorPaletteSelector.CUSTOM_PALETTE_NAME

    fake_dialog = Mock()
    fake_dialog.exec.return_value = False
    monkeypatch.setattr(
        plot_color_palette_selector,
        "CustomPaletteDialog",
        Mock(return_value=fake_dialog),
    )

    selector._open_custom_palette_dialog()

    assert selector.findText(custom_name, Qt.MatchFlag.MatchFixedString) == -1
    connection_point.assert_not_called()
