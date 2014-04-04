#  Copyright (C) 2014  Statoil ASA, Norway.
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
from PyQt4.QtCore import Qt, QSize

from PyQt4.QtGui import  QFormLayout, QWidget, QLineEdit, QToolButton, QHBoxLayout, QFileDialog, QComboBox
from ert_gui.ide.keywords.definitions import RangeStringArgument
from ert_gui.models.connectors import EnsembleSizeModel
from ert_gui.models.connectors.init import CaseSelectorModel
from ert_gui.models.connectors.plot import PlotSettingsModel
from ert_gui.tools.export.export_realizations_model import ExportRealizationsModel
from ert_gui.widgets import util
from ert_gui.widgets.string_box import StringBox


class ExportPanel(QWidget):
    def __init__(self):
        QWidget.__init__(self)


        self.setMinimumWidth(750)
        self.setMinimumHeight(500)

        self.setWindowTitle("Export data")
        self.activateWindow()

        self.__current_case = CaseSelectorModel().getCurrentChoice()

        layout = QFormLayout()

        active_realizations_model = ExportRealizationsModel(EnsembleSizeModel().getValue())
        self.active_realizations_field = StringBox(active_realizations_model, "Active realizations", "config/simulation/active_realizations")
        self.active_realizations_field.setValidator(RangeStringArgument())
        layout.addRow(self.active_realizations_field.getLabel(), self.active_realizations_field)


        file_name_button= QToolButton()
        file_name_button.setText("Browse")
        file_name_button.clicked.connect(self.selectFileDirectory)


        self.__file_name = QLineEdit()
        self.__file_name.setEnabled(False)
        self.__file_name.setMinimumWidth(250)

        file_name_layout = QHBoxLayout()
        file_name_layout.addWidget(self.__file_name)
        file_name_layout.addWidget(file_name_button)

        layout.addRow("Select directory to save files:", file_name_layout)

        self.__file_type_combo = QComboBox()
        self.__file_type_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.__file_type_combo.addItem("Eclipse GRDECL")
        self.__file_type_combo.addItem("RMS roff")
        layout.addRow("Select file format:",self.__file_type_combo)

        export_button= QToolButton()
        export_button.setText("Export")
        export_button.clicked.connect(self.export)
        export_button.setIcon(util.resourceIcon("ide/table_export"))
        export_button.setIconSize(QSize(32, 32))
        export_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        layout.addRow("",export_button)

        self.setLayout(layout)


    def selectFileDirectory(self):

        directory = QFileDialog().getExistingDirectory(self, "Directory", PlotSettingsModel().getDefaultPlotPath(), QFileDialog.ShowDirsOnly)
        self.__file_name.setText(str(directory))


    def export(self):
        pass


