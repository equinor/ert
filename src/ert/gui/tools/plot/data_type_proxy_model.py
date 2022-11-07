from typing import TYPE_CHECKING

from qtpy.QtCore import QSortFilterProxyModel, Qt

if TYPE_CHECKING:
    from .data_type_keys_list_model import DataTypeKeysListModel


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

    def sourceModel(self) -> "DataTypeKeysListModel":
        return QSortFilterProxyModel.sourceModel(self)

    def setFilterOnMetadata(self, key, value, visible):
        if key not in self._metadata_filters:
            self._metadata_filters[key] = {}

        self._metadata_filters[key][value] = visible
        self.invalidateFilter()
