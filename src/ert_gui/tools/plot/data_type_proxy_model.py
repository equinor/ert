#  Copyright (C) 2014  Equinor ASA, Norway.
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
from qtpy.QtCore import Qt, QSortFilterProxyModel


from ert_gui.tools.plot import DataTypeKeysListModel


class DataTypeProxyModel(QSortFilterProxyModel):
    def __init__(self, parent, model):
        QSortFilterProxyModel.__init__(self, parent)

        self.__show_summary_keys = True
        self.__show_block_keys = True
        self.__show_gen_kw_keys = True
        self.__show_gen_data_keys = True
        self.__show_custom_pca_keys = True
        self._metadata_filters = {}
        self.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.setSourceModel(model)

    def filterAcceptsRow(self, index, q_model_index):
        show = QSortFilterProxyModel.filterAcceptsRow(self, index, q_model_index)

        if show:
            source_model = self.sourceModel()
            source_index = source_model.index(index, 0, q_model_index)
            key = source_model.itemAt(source_index)

            for meta_key, values in self._metadata_filters.items():
                for value, visible in values.items():
                    if (
                        not visible
                        and meta_key in key["metadata"]
                        and key["metadata"][meta_key] == value
                    ):
                        show = False

        return show

    def sourceModel(self):
        """@rtype: DataTypeKeysListModel"""
        return QSortFilterProxyModel.sourceModel(self)

    def setFilterOnMetadata(self, key, value, visible):
        if not key in self._metadata_filters:
            self._metadata_filters[key] = {}

        self._metadata_filters[key][value] = visible
        self.invalidateFilter()
