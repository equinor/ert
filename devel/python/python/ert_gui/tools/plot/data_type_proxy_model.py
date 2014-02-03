#  Copyright (C) 2014  Statoil ASA, Norway.
#   
#  The file 'data_type_proxy_model.py' is part of ERT - Ensemble based Reservoir Tool.
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

from PyQt4.QtCore import Qt
from PyQt4.QtGui import QSortFilterProxyModel
from ert_gui.tools.plot import DataTypeKeysListModel


class DataTypeProxyModel(QSortFilterProxyModel):

    def __init__(self, model , parent=None):
        QSortFilterProxyModel.__init__(self, parent)
        self.__show_summary_keys = True
        self.__show_block_keys = True

        self.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.setSourceModel(model)

    def filterAcceptsRow(self, index, q_model_index):
        show = QSortFilterProxyModel.filterAcceptsRow(self, index, q_model_index)

        if show:
            source_index = self.sourceModel().index(index, 0, q_model_index)
            key = self.sourceModel().itemAt(source_index)

            summary_key = self.sourceModel().isSummaryKey(key)
            if not self.__show_summary_keys and summary_key:
                show = False

            block_key = self.sourceModel().isBlockKey(key)
            if not self.__show_block_keys and block_key:
                show = False

        return show

    def sourceModel(self):
        """ @rtype: DataTypeKeysListModel """
        return QSortFilterProxyModel.sourceModel(self)

    def setShowSummaryKeys(self, visible):
        self.__show_summary_keys = visible
        self.invalidateFilter()

    def setShowBlockKeys(self, visible):
        self.__show_block_keys = visible
        self.invalidateFilter()