from qtpy.QtCore import Qt

from ert.gui.ertwidgets.checklist import CheckList
from ert.gui.ertwidgets.models.selectable_list_model import SelectableListModel


def test_checklist(qtbot):
    checklist = CheckList(SelectableListModel(items=["1", "2", "3"]))
    qtbot.addWidget(checklist)

    qtbot.mouseClick(checklist._checkAllButton, Qt.LeftButton)

    for item in checklist._model.getList():
        assert checklist._model.isValueSelected(item)

    qtbot.mouseClick(checklist._uncheckAllButton, Qt.LeftButton)

    for item in checklist._model.getList():
        assert not checklist._model.isValueSelected(item)
