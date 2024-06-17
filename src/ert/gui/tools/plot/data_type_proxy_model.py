from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from qtpy.QtCore import QModelIndex, QObject, QSortFilterProxyModel, Qt

if TYPE_CHECKING:
    from .data_type_keys_list_model import DataTypeKeysListModel


class DataTypeProxyModel(QSortFilterProxyModel):
    def __init__(self, parent: Optional[QObject], model: DataTypeKeysListModel) -> None:
        QSortFilterProxyModel.__init__(self, parent)

        self.__show_summary_keys = True
        self.__show_block_keys = True
        self.__show_gen_kw_keys = True
        self.__show_gen_data_keys = True
        self.__show_custom_pca_keys = True
        self._metadata_filters: dict[str, dict[str, bool]] = {}
        self.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.setSourceModel(model)

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        show = QSortFilterProxyModel.filterAcceptsRow(self, source_row, source_parent)

        if show:
            source_model = self.sourceModel()
            source_index = source_model.index(source_row, 0, source_parent)
            key = source_model.itemAt(source_index)
            assert key is not None

            for meta_key, values in self._metadata_filters.items():
                for value, visible in values.items():
                    if (
                        not visible
                        and meta_key in key.metadata
                        and key.metadata[meta_key] == value
                    ):
                        show = False

        return show

    def sourceModel(self) -> DataTypeKeysListModel:
        return QSortFilterProxyModel.sourceModel(self)  # type: ignore

    def setFilterOnMetadata(self, key: str, value: str, visible: bool) -> None:
        if key not in self._metadata_filters:
            self._metadata_filters[key] = {}

        self._metadata_filters[key][value] = visible
        self.invalidateFilter()
