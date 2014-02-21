#  Copyright (C) 2014  Statoil ASA, Norway.
#   
#  The file 'export_plot_widget.py' is part of ERT - Ensemble based Reservoir Tool.
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

from PyQt4.QtCore import pyqtSignal, Qt
from PyQt4.QtGui import QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QToolButton
from ert_gui.widgets import util


class ExportPlotWidget(QWidget):
    exportButtonPressed = pyqtSignal()

    def __init__(self):
        QWidget.__init__(self)

        layout = QVBoxLayout()
        add_button_layout = QHBoxLayout()
        export_button = QToolButton()
        export_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        export_button.setText("Export Plot")
        export_button.setIcon(util.resourceIcon("ide/small/chart_curve_go"))
        export_button.clicked.connect(self.exportButtonPressed.emit)
        add_button_layout.addStretch()
        add_button_layout.addWidget(export_button)
        add_button_layout.addStretch()
        layout.addLayout(add_button_layout)
        layout.addStretch()

        self.setLayout(layout)
