from uuid import uuid4

import numpy as np
from PyQt6.QtWidgets import QTableWidget
from pytestqt.qtbot import QtBot

from ert.analysis.event import DataSection
from ert.gui.experiments.view import UpdateWidget
from ert.run_models.event import RunModelDataEvent


def test_update_widget(qtbot: QtBot):
    event = RunModelDataEvent(
        iteration=0,
        name="test",
        run_id=uuid4(),
        data=DataSection(header=["a", "b"], data=np.array([[42, 2], [3, 4]])),
    )
    widget = UpdateWidget(event.iteration)
    widget.show()
    qtbot.addWidget(widget)
    widget.add_table(event)
    table = widget.findChild(QTableWidget, "CSV_test")

    assert table is not None
    assert (table.columnCount(), table.rowCount()) == (2, 2)
    assert table.item(1, 1).text() == "4"
