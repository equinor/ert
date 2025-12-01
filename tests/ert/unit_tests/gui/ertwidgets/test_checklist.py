from PyQt6.QtCore import Qt

from ert.gui.ertwidgets import CheckList, SelectableListModel


def test_checklist(qtbot):
    checklist = CheckList(SelectableListModel(items=["1", "2", "3"]))
    qtbot.addWidget(checklist)

    qtbot.mouseClick(checklist._checkAllButton, Qt.MouseButton.LeftButton)

    for item in checklist._model.getList():
        assert checklist._model.isValueSelected(item)

    qtbot.mouseClick(checklist._uncheckAllButton, Qt.MouseButton.LeftButton)

    for item in checklist._model.getList():
        assert not checklist._model.isValueSelected(item)
