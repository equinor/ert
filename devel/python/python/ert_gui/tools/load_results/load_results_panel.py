#  Copyright (C) 2014  Statoil ASA, Norway.
#
#  The file 'load_results_panel.py' is part of ERT - Ensemble based Reservoir Tool.
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
from PyQt4.QtGui import QWidget, QFormLayout, QComboBox, QLineEdit
from ert_gui.ide.keywords.definitions import RangeStringArgument
from ert_gui.models.connectors import EnsembleSizeModel
from ert_gui.models.connectors.init import CaseSelectorModel
from ert_gui.models.connectors.load_results import LoadResultsModel
from ert_gui.tools.load_results import LoadResultsRealizationsModel
from ert_gui.tools.manage_cases.all_cases_model import AllCasesModel
from ert_gui.widgets.string_box import StringBox


class LoadResultsPanel(QWidget):

    def __init__(self):
        QWidget.__init__(self)

        self.setMinimumWidth(500)
        self.setMinimumHeight(200)
        self.__dynamic = False

        self.setWindowTitle("Load results manually")
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


        self.__active_realizations_model = LoadResultsRealizationsModel(EnsembleSizeModel().getValue())
        self.__active_realizations_field = StringBox(self.__active_realizations_model, "Active realizations", "config/simulation/active_realizations")
        self.__active_realizations_field.setValidator(RangeStringArgument())
        layout.addRow(self.__active_realizations_field.getLabel(), self.__active_realizations_field)


        self.__iterations_count = LoadResultsModel().getIterationCount()

        self.__iterations = QLineEdit()
        self.__iterations.setMinimumWidth(250)

        layout.addRow("Iteration to load:", self.__iterations)

        self.setLayout(layout)

    def load(self):
        pass
