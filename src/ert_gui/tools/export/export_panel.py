#  Copyright (C) 2014  Equinor ASA, Norway.
#
#  The file 'export_panel.py' is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.
import sys

from qtpy.QtCore import QDir, Signal
from qtpy.QtWidgets import (
    QFormLayout,
    QWidget,
    QLineEdit,
    QToolButton,
    QHBoxLayout,
    QFileDialog,
)


class ExportPanel(QWidget):
    updateExportButton = Signal(str, bool)
    runExport = Signal(dict)

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.setMinimumWidth(500)
        self.setMinimumHeight(200)
        self._dynamic = False

        self.setWindowTitle("Export data")
        self.activateWindow()

        layout = QFormLayout()

        self._column_keys_input = QLineEdit()
        self._column_keys_input.setMinimumWidth(250)
        self._column_keys_input.setText("*")
        layout.addRow("Columns to export:", self._column_keys_input)

        self._time_index_input = QLineEdit()
        self._time_index_input.setMinimumWidth(250)
        self._time_index_input.setText("raw")
        layout.addRow("Time index:", self._time_index_input)

        file_name_button = QToolButton()
        file_name_button.setText("Browse")
        file_name_button.clicked.connect(self.selectFileDirectory)
        self._defaultPath = QDir.currentPath() + "/export.csv"
        self._file_name = QLineEdit()
        self._file_name.setEnabled(False)
        self._file_name.setText(self._defaultPath)
        self._file_name.setMinimumWidth(250)

        file_name_layout = QHBoxLayout()
        file_name_layout.addWidget(self._file_name)
        file_name_layout.addWidget(file_name_button)
        layout.addRow("Select directory to save files to:", file_name_layout)

        self.setLayout(layout)

    def selectFileDirectory(self):
        directory = QFileDialog(self).getExistingDirectory(
            self, "Directory", self._file_name.text(), QFileDialog.ShowDirsOnly
        )
        if str(directory).__len__() > 0:
            self._file_name.setText(str(directory))

    def export(self):

        path = self._file_name.text()
        time_index = self._time_index_input.text()
        column_keys = self._column_keys_input.text()
        values = {
            "output_file": path,
            "time_index": time_index,
            "column_keys": column_keys,
        }
        self.runExport.emit(values)

    def selectFileDirectory(self):
        directory = QFileDialog().getExistingDirectory(
            self, "Directory", self._file_name.text(), QFileDialog.ShowDirsOnly
        )
        if str(directory).__len__() > 0:
            self._file_name.setText(str(directory))
