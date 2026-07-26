from PyQt6.QtCore import Qt

from ert.gui.plotting.customization_dialog.style_chooser import (
    STYLESET_AREA,
    StyleChooser,
)


def test_that_compact_style_chooser_uses_symbols_without_changing_full_labels(qtbot):
    full_size_chooser = StyleChooser()
    compact_chooser = StyleChooser(compact=True)
    qtbot.addWidget(full_size_chooser)
    qtbot.addWidget(compact_chooser)

    full_size_labels = [
        full_size_chooser.line_chooser.itemText(index)
        for index in range(full_size_chooser.line_chooser.count())
    ]
    compact_labels = [
        compact_chooser.line_chooser.itemText(index)
        for index in range(compact_chooser.line_chooser.count())
    ]

    assert full_size_labels == ["Off", "Solid", "Dashed", "Dotted", "Dash dotted"]
    assert compact_labels == ["Off", "-", "--", "..", "-."]

    dotted_index = compact_chooser.line_chooser.findData(":")
    assert (
        compact_chooser.line_chooser.itemData(dotted_index, Qt.ItemDataRole.ToolTipRole)
        == "Dotted"
    )
    assert (
        compact_chooser.line_chooser.itemData(
            dotted_index, Qt.ItemDataRole.AccessibleTextRole
        )
        == "Dotted"
    )

    circle_index = compact_chooser.marker_chooser.findData("o")
    assert compact_chooser.marker_chooser.itemText(circle_index) == "○"
    assert (
        compact_chooser.marker_chooser.itemData(
            circle_index, Qt.ItemDataRole.ToolTipRole
        )
        == "Circle"
    )
    assert (
        compact_chooser.marker_chooser.itemData(
            circle_index, Qt.ItemDataRole.AccessibleTextRole
        )
        == "Circle"
    )


def test_that_compact_area_style_disables_marker_selection(qtbot):
    chooser = StyleChooser(line_style_set=STYLESET_AREA, compact=True)
    qtbot.addWidget(chooser)

    chooser.line_chooser.setCurrentIndex(chooser.line_chooser.findData("#"))
    assert not chooser.marker_chooser.isEnabled()

    chooser.line_chooser.setCurrentIndex(chooser.line_chooser.findData("-"))
    assert chooser.marker_chooser.isEnabled()
