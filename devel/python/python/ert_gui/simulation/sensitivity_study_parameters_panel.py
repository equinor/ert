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

from collections import namedtuple

from PyQt4.QtCore import Qt
from PyQt4.QtGui import QWidget, QHBoxLayout, QTableWidget, QTableWidgetItem, \
    QCheckBox, QHeaderView, QLabel


class SensitivityStudyParametersPanel(QWidget):

    Column = namedtuple("Column_tuple", "index header")

    columns = {"name"        : Column(index = 0, header = "Parameter Name"),
               "is_active"   : Column(index = 1, header = "Include"),
               "const_value" : Column(index = 2, header = "Constant Value")}
    
    column_list = ["name", "is_active", "const_value"]

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        # For test purposes. Will be read from config later.
        # model = SensitivityStudyParametersModel()
        parameters = ["para_1", "para_2", "para_3"] # model.getParameters()
        is_active = [True, True, False] # model...
        constant_value = [0, 1, 2] # model...
        n_parameters = len(parameters)
        
        layout = QHBoxLayout()

        table = QTableWidget(n_parameters, len(self.columns), self)
        table.verticalHeader().setResizeMode(QHeaderView.Fixed)
        table.verticalHeader().hide()
        
        headers = [self.columns[col_id].header for col_id in self.column_list]
        table.setHorizontalHeaderLabels(headers)
        # table.setVerticalHeaderLabels(parameters)
        table.setMinimumWidth(400)

        for row in range(n_parameters):
            param_name_widget = QLabel(parameters[row])
            table.setCellWidget(row, self.columns["name"].index, param_name_widget)
          
            const_value_item = QTableWidgetItem(str(constant_value[row]))
            const_value_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            const_value_item
            table.setItem(row, self.columns["const_value"].index, const_value_item)

            is_active_widget = QWidget()
            is_active_layout = QHBoxLayout(is_active_widget)
            is_active_checkbox = QCheckBox()
            is_active_checkbox.setChecked(is_active[row])
            is_active_layout.addWidget(is_active_checkbox)
            is_active_layout.setAlignment(Qt.AlignCenter)
            is_active_layout.setContentsMargins(0, 0, 0, 0)
            is_active_widget.setLayout(is_active_layout)
            table.setCellWidget(row, self.columns["is_active"].index, is_active_widget)
            if not is_active[row]:
                const_value_item.setFlags(const_value_item.flags() & (not Qt.ItemIsEditable))


        layout.addWidget(table)

        self.setLayout(layout)
        self.blockSignals(False)

    def ValueChanged(self):
        return False
