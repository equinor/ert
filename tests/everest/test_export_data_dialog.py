import os
from pathlib import Path

import pytest
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from qtpy.QtCore import Qt, QTimer

from everest.config.everest_config import EverestConfig
from ieverest import IEverest
from ieverest.widgets.export_data_dialog import ExportDataDialog


@pytest.mark.ui_test
@pytest.mark.xdist_group(name="starts_everest")
def test_export_data_dialog(qtbot, monkeypatch, change_to_tmpdir):
    ieverest = IEverest()

    config = EverestConfig.with_defaults()
    dialog = ExportDataDialog(config)
    dialog.show()

    dialog.keywords_txt.setText("keyword")
    dialog.batches_txt.setText("1,2")

    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda parent, caption, dir, filter: (dir, "")
    )

    def handle_message_box():
        qtbot.waitUntil(lambda: ieverest._gui.findChild(QMessageBox) is not None)
        widget = ieverest._gui.findChild(QMessageBox)
        qtbot.mouseClick(widget.button(QMessageBox.Ok), Qt.MouseButton.LeftButton)
        assert "Export completed successfully!" in widget.text()

    QTimer.singleShot(100, handle_message_box)
    dialog.export_btn.click()

    assert Path(os.path.join(config.output_dir, "export.csv")).exists()
