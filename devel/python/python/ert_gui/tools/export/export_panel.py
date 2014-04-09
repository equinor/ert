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
from PyQt4.QtCore import Qt, QSize, QDir

from PyQt4.QtGui import  QFormLayout, QWidget, QLineEdit, QToolButton, QHBoxLayout, QFileDialog, QComboBox
from ert.enkf import ErtImplType, EnkfFieldFileFormatEnum, EnkfStateType
from ert_gui.ide.keywords.definitions import RangeStringArgument
from ert_gui.models.connectors import EnsembleSizeModel
from ert_gui.models.connectors.export import ExportKeywordModel, ExportModel
from ert_gui.models.connectors.init import CaseSelectorModel
from ert_gui.tools.export import ExportRealizationsModel
from ert_gui.widgets import util
from ert_gui.widgets.string_box import StringBox


class ExportPanel(QWidget):
    def __init__(self):
        QWidget.__init__(self)

        self.setMinimumWidth(500)
        self.setMinimumHeight(200)
        self.__dynamic = False

        self.setWindowTitle("Export data")
        self.activateWindow()

        layout = QFormLayout()

        self.__keywords = ExportKeywordModel().getKeylistFromImplType(ErtImplType.FIELD)
        self.__fields_keyword = QComboBox()
        self.__fields_keyword.addItems(self.__keywords)
        layout.addRow("Select keyword:",self.__fields_keyword)

        self.__current_case = CaseSelectorModel().getCurrentChoice()

        self.__active_realizations_model = ExportRealizationsModel(EnsembleSizeModel().getValue())
        self.__active_realizations_field = StringBox(self.__active_realizations_model, "Active realizations", "config/simulation/active_realizations")
        self.__active_realizations_field.setValidator(RangeStringArgument())
        layout.addRow(self.__active_realizations_field.getLabel(), self.__active_realizations_field)

        file_name_button= QToolButton()
        file_name_button.setText("Browse")
        file_name_button.clicked.connect(self.selectFileDirectory)


        self.__file_name = QLineEdit()
        self.__file_name.setEnabled(False)
        self.__file_name.setText(QDir.currentPath())
        self.__file_name.setMinimumWidth(250)

        file_name_layout = QHBoxLayout()
        file_name_layout.addWidget(self.__file_name)
        file_name_layout.addWidget(file_name_button)

        layout.addRow("Select directory to save files:", file_name_layout)

        self.__file_type_model = ["Eclipse GRDECL", "RMS roff"]
        self.__file_type_combo = QComboBox()
        self.__file_type_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.__file_type_combo.addItems(self.__file_type_model)
        layout.addRow("Select file format:",self.__file_type_combo)

        self.__report_step = QLineEdit()
        layout.addRow("Report step:", self.__report_step)

        export_button= QToolButton()
        export_button.setText("Export")
        export_button.clicked.connect(self.export)
        export_button.setIcon(util.resourceIcon("ide/table_export"))
        export_button.setIconSize(QSize(32, 32))
        export_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)


        layout.addRow("",export_button)

        self.setLayout(layout)
        self.__fields_keyword.currentIndexChanged.connect(self.keywordSelected)
        self.keywordSelected()


    def selectFileDirectory(self):

        directory = QFileDialog().getExistingDirectory(self, "Directory", QDir.currentPath(), QFileDialog.ShowDirsOnly)
        self.__file_name.setText(str(directory))


    def export(self):
        report_step = 0
        if self.__dynamic:
            report_step = self.__report_step.text()
        keyword = self.__keywords[self.__fields_keyword.currentIndex()]

        path = self.__file_name.text()
        if QDir(path).exists():
            file_name  = str(path + "/" + keyword +"_%d")
            iactive = self.__active_realizations_model.getActiveRealizationsMask()

            file_type_key = self.__file_type_model[self.__file_type_combo.currentIndex()]

            if file_type_key == "Eclipse GRDECL":
                file_type = EnkfFieldFileFormatEnum.ECL_GRDECL_FILE
            else:
                file_type = EnkfFieldFileFormatEnum.RMS_ROFF_FILE

            state = EnkfStateType.FORECAST
            export_model = ExportModel()
            export_model.exportField(keyword, file_name, iactive, file_type, report_step, state)



    def keywordSelected(self):
        key = self.__keywords[self.__fields_keyword.currentIndex()]
        self.__dynamic = ExportKeywordModel().isDynamicField(key)
        self.__report_step.setVisible(self.__dynamic)
        self.layout().labelForField(self.__report_step).setVisible(self.__dynamic)
