from ert.gui.plotting.utils.plot_color_palettes import MINIMUM_COLOR_CYCLE_LENGTH
from ert.gui.plotting.widgets.plot_controls.custom_palette_dialog import (
    CustomPaletteDialog,
)


def test_that_dialog_seeds_color_boxes_from_the_given_cycle(qtbot):
    color_cycle = [("#ff0000", 1.0), ("#00ff00", 1.0), ("#0000ff", 1.0)]
    dialog = CustomPaletteDialog(color_cycle)
    qtbot.addWidget(dialog)

    returned = dialog.get_color_cycle()

    assert returned[:3] == color_cycle


def test_that_dialog_exposes_default_amount_of_color_boxes(qtbot):
    dialog = CustomPaletteDialog([])
    qtbot.addWidget(dialog)

    assert len(dialog.get_color_cycle()) == MINIMUM_COLOR_CYCLE_LENGTH
