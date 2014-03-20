#  Copyright (C) 2013  Statoil ASA, Norway.
#
#  The file 'analysis_module_variables_panel.py' is part of ERT - Ensemble based Reservoir Tool.
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

from PyQt4.QtGui import QDoubleSpinBox, QWidget, QFormLayout
from ert_gui.models.connectors.run.analysis_module_variables_model import AnalysisModuleVariablesModel



class AnalysisModuleVariablesPanel(QWidget):

    def __init__(self, analysis_module, parent=None):
        QWidget.__init__(self, parent)

        layout = QFormLayout()
        variable_names = AnalysisModuleVariablesModel().getVariableNames(analysis_module)

        for variable in variables_name:
            variable_type = AnalysisModuleVariablesModel().getVariableType(variable)

            spinner = QDoubleSpinBox()
            spinner.setMinimumWidth(75)
            layout.addRow(variable, spinner)


        self.setLayout(layout)