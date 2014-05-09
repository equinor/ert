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
import os
from PyQt4.QtCore import QDir

from PyQt4.QtGui import  QFormLayout, QWidget, QLineEdit, QToolButton, QHBoxLayout, QFileDialog, QComboBox, QMessageBox
from ert.enkf import EnkfFieldFileFormatEnum, EnkfStateType
from ert_gui.ide.keywords.definitions import RangeStringArgument
from ert_gui.models.connectors import EnsembleSizeModel
from ert_gui.models.connectors.export import ExportKeywordModel, ExportModel
from ert_gui.models.connectors.init import CaseSelectorModel
from ert_gui.tools.export import ExportRealizationsModel
from ert_gui.tools.manage_cases.all_cases_model import AllCasesModel
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
        current_case = CaseSelectorModel().getCurrentChoice()

        self.__case_model = AllCasesModel()
        self.__case_combo = QComboBox()
        self.__case_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLength)
        self.__case_combo.setMinimumContentsLength(20)
        self.__case_combo.setModel(self.__case_model)
        self.__case_combo.setCurrentIndex(self.__case_model.indexOf(current_case))
        layout.addRow("Select case:",self.__case_combo)

        self.__export_keyword_model = ExportKeywordModel()

        self.__kw_model = self.__export_keyword_model.getKeyWords()
        self.__keywords = QComboBox()
        self.__keywords.addItems(self.__kw_model)
        layout.addRow("Select keyword:",self.__keywords)

        self.__active_realizations_model = ExportRealizationsModel(EnsembleSizeModel().getValue())
        self.__active_realizations_field = StringBox(self.__active_realizations_model, "Active realizations", "config/simulation/active_realizations")
        self.__active_realizations_field.setValidator(RangeStringArgument())
        layout.addRow(self.__active_realizations_field.getLabel(), self.__active_realizations_field)

        file_name_button= QToolButton()
        file_name_button.setText("Browse")
        file_name_button.clicked.connect(self.selectFileDirectory)

        self.__file_name = QLineEdit()
        self.__file_name.setEnabled(False)
        self.__file_name.setText(QDir.currentPath()+"/export")
        self.__file_name.setMinimumWidth(250)

        file_name_layout = QHBoxLayout()
        file_name_layout.addWidget(self.__file_name)
        file_name_layout.addWidget(file_name_button)
        layout.addRow("Select directory to save files to:", file_name_layout)

        self.__gen_kw_file_types = ["Parameter list", "Template based"]
        self.__field_kw_file_types = ["Eclipse GRDECL", "RMS roff"]

        self.__file_type_model = self.__field_kw_file_types
        self.__file_type_combo = QComboBox()
        self.__file_type_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.__file_type_combo.addItems(self.__file_type_model)
        layout.addRow("Select file format:",self.__file_type_combo)

        self.__report_step = QLineEdit()
        layout.addRow("Report step:", self.__report_step)

        self.setLayout(layout)
        self.__keywords.currentIndexChanged.connect(self.keywordSelected)
        self.keywordSelected()

    def selectFileDirectory(self):
        directory = QFileDialog().getExistingDirectory(self, "Directory", QDir.currentPath(), QFileDialog.ShowDirsOnly)
        self.__file_name.setText(str(directory))

    def updateFileExportType(self):
        keyword = self.__kw_model[self.__keywords.currentIndex()]
        self.__file_type_combo.clear()
        if self.__export_keyword_model.isGenKw(keyword):
            self.__file_type_model = self.__gen_kw_file_types
        else:
            self.__file_type_model = self.__field_kw_file_types

        self.__file_type_combo.addItems(self.__file_type_model)

    def export(self):
        report_step = 0
        if self.__dynamic:
            report_step = self.__report_step.text()
        keyword = self.__kw_model[self.__keywords.currentIndex()]
        all_cases = self.__case_model.getAllItems()
        selected_case  = all_cases[self.__case_combo.currentIndex()]

        file_name = self.createExportFilNameMask(keyword, selected_case)

        iactive = self.__active_realizations_model.getActiveRealizationsMask()

        file_type_key = self.__file_type_model[self.__file_type_combo.currentIndex()]
        state = EnkfStateType.FORECAST

        if self.__export_keyword_model.isFieldKw(keyword):
            self.exportField(keyword, file_name, iactive, file_type_key, report_step, state, selected_case)
        elif self.__export_keyword_model.isGenKw(keyword):
            self.exportGenKw(keyword, file_name, iactive, file_type_key, report_step, state, selected_case)


    def exportGenKw(self, keyword, file_name, iactive, file_type_key, report_step, state, selected_case):
        ExportModel().exportGenKw(keyword, file_name, iactive, file_type_key, report_step, state, selected_case)


    def exportField(self, keyword, file_name, iactive, file_type_key, report_step, state, selected_case):
        if file_type_key == "Eclipse GRDECL":
            file_type = EnkfFieldFileFormatEnum.ECL_GRDECL_FILE
        else:
            file_type = EnkfFieldFileFormatEnum.RMS_ROFF_FILE

        result = ExportModel().exportField(keyword, file_name, iactive, file_type, report_step, state, selected_case)
        if not result:
            QMessageBox.warning(self, "Warning",'''Something did not work!''',QMessageBox.Ok);

    def createExportFilNameMask(self, keyword, current_case):
        path = self.__file_name.text()
        impl_type = None

        if self.__export_keyword_model.isFieldKw(keyword):
            impl_type = ExportKeywordModel().getImplementationType(keyword)
        elif self.__export_keyword_model.isGenKw(keyword):
            impl_type = "Gen_Kw"

        path = str(path) + "/" + str(current_case) + "/" + str(impl_type) + "/" + str(keyword)

        if not QDir(path).exists():
            os.makedirs(path);

        return path

    def keywordSelected(self):
        self.updateFileExportType()
        key = self.__kw_model[self.__keywords.currentIndex()]
        if self.__export_keyword_model.isFieldKw(key):
            self.__dynamic = ExportKeywordModel().isDynamicField(key)
        else:
            self.__dynamic = False

        self.__report_step.setVisible(self.__dynamic)
        self.layout().labelForField(self.__report_step).setVisible(self.__dynamic)
