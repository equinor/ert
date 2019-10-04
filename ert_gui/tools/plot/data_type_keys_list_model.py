try:
  from PyQt4.QtCore import QAbstractItemModel, QModelIndex, Qt, QVariant
  from PyQt4.QtGui import QColor
except ImportError:
  from PyQt5.QtCore import QAbstractItemModel, QModelIndex, Qt, QVariant
  from PyQt5.QtGui import QColor

from ert_gui.ertwidgets import resourceIcon
from ert_shared import ERT


class DataTypeKeysListModel(QAbstractItemModel):
    DEFAULT_DATA_TYPE = QColor(255, 255, 255)
    HAS_OBSERVATIONS = QColor(237, 218, 116)
    GROUP_ITEM = QColor(64, 64, 64)

    def __init__(self):
        QAbstractItemModel.__init__(self)        
        self.__icon = resourceIcon("ide/small/bullet_star")    

    def index(self, row, column, parent=None, *args, **kwargs):
        return self.createIndex(row, column)

    def parent(self, index=None):
        return QModelIndex()

    def rowCount(self, parent=None, *args, **kwargs):
        return len(ERT.enkf_facade.get_all_data_type_keys())

    def columnCount(self, QModelIndex_parent=None, *args, **kwargs):
        return 1

    def data(self, index, role=None):
        assert isinstance(index, QModelIndex)

        if index.isValid():
            items = ERT.enkf_facade.get_all_data_type_keys()
            row = index.row()
            item = items[row]

            if role == Qt.DisplayRole:
                return item
            elif role == Qt.BackgroundRole:
                if ERT.enkf_facade.is_key_with_observations(item):                
                    return self.HAS_OBSERVATIONS

        return QVariant()

    def itemAt(self, index):
        assert isinstance(index, QModelIndex)

        if index.isValid():
            row = index.row()
            return ERT.enkf_facade.get_data_type_key(row)

        return None


    def isSummaryKey(self, key):
        return ERT.enkf_facade.is_summary_key(key)

    def isBlockKey(self, key):
        return False

    def isGenKWKey(self, key):
        return ERT.enkf_facade.is_gen_kw_key(key)

    def isGenDataKey(self, key):
        return ERT.enkf_facade.is_gen_data_key(key)

    def isCustomKwKey(self, key):
        return ERT.enkf_facade.is_custom_kw_key(key)

    def isCustomPcaKey(self, key):
        return False
