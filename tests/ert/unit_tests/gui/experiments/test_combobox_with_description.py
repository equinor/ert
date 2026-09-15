import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPixmap
from PyQt6.QtWidgets import QStyle, QStyleOptionViewItem
from pytestqt.qtbot import QtBot

from ert.gui.experiments.combobox_with_description import (
    DESCRIPTION_ROLE,
    GROUP_TITLE_ROLE,
    QComboBoxWithDescription,
)


def test_that_models_are_inserted_under_one_header_per_group(qtbot: QtBot) -> None:
    combo = QComboBoxWithDescription()
    qtbot.addWidget(combo)
    selections = []
    combo.currentTextChanged.connect(selections.append)

    combo.addDescriptionItem("First", "First description", "Evaluation")
    combo.addDescriptionItem("Second", "Second description", "Update")
    row = combo.addDescriptionItem("Third", "Third description", "Evaluation")

    assert [combo.itemText(i) for i in range(combo.count())] == [
        "Evaluation",
        "First",
        "Third",
        "Update",
        "Second",
    ]
    assert combo.itemText(row) == "Third"
    assert combo.itemData(row, DESCRIPTION_ROLE) == "Third description"
    assert combo.itemData(row, GROUP_TITLE_ROLE) is None
    assert combo.currentText() == "First"
    assert selections == ["First"]
    for row in (0, 3):
        assert combo.model().index(row, 0).flags() == Qt.ItemFlag.NoItemFlags


def test_that_ungrouped_models_have_no_header(qtbot: QtBot) -> None:
    combo = QComboBoxWithDescription()
    qtbot.addWidget(combo)
    combo.addDescriptionItem("First", "Description")
    combo.addDescriptionItem("Second", "Description")

    assert combo.count() == 2
    assert combo.currentText() == "First"
    assert all(combo.itemData(i, GROUP_TITLE_ROLE) is None for i in range(2))


def test_that_keyboard_navigation_skips_group_headers(qtbot: QtBot) -> None:
    combo = QComboBoxWithDescription()
    qtbot.addWidget(combo)
    combo.addDescriptionItem("First", "Description", "Evaluation")
    combo.addDescriptionItem("Second", "Description", "Update")
    combo.show()

    qtbot.keyClick(combo, Qt.Key.Key_Down)
    assert combo.currentText() == "Second"
    qtbot.keyClick(combo, Qt.Key.Key_Up)
    assert combo.currentText() == "First"
    qtbot.keyClick(combo, Qt.Key.Key_Up)
    assert combo.currentText() == "First"


def test_that_clicking_a_header_does_not_select_it(qtbot: QtBot) -> None:
    combo = QComboBoxWithDescription()
    qtbot.addWidget(combo)
    combo.addDescriptionItem("First", "Description", "Evaluation")
    combo.addDescriptionItem("Second", "Description", "Update")
    combo.show()
    combo.showPopup()
    view = combo.view()
    header_index = combo.model().index(2, 0)
    header_center = view.visualRect(header_index).center()

    qtbot.mouseMove(view.viewport(), header_center)
    qtbot.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=header_center)

    assert combo.currentText() == "First"
    assert not view.selectionModel().isSelected(header_index)
    assert view.isVisible()

    item_center = view.visualRect(combo.model().index(3, 0)).center()
    qtbot.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=item_center)
    assert combo.currentText() == "Second"


@pytest.mark.parametrize(
    "state", [QStyle.StateFlag.State_MouseOver, QStyle.StateFlag.State_Selected]
)
def test_that_headers_are_not_highlighted(
    qtbot: QtBot, state: QStyle.StateFlag
) -> None:
    combo = QComboBoxWithDescription()
    qtbot.addWidget(combo)
    combo.addDescriptionItem("First", "Description", "Evaluation")
    index = combo.model().index(0, 0)
    option = QStyleOptionViewItem()
    option.widget = combo.view()
    option.rect.setSize(combo.itemDelegate().sizeHint(option, index))
    images = []
    for extra_state in (QStyle.StateFlag.State_None, state):
        option.state = QStyle.StateFlag.State_Enabled | extra_state
        pixmap = QPixmap(option.rect.size())
        pixmap.fill(Qt.GlobalColor.white)
        painter = QPainter(pixmap)
        combo.itemDelegate().paint(painter, option, index)
        painter.end()
        images.append(pixmap.toImage())

    assert images[0] == images[1]
