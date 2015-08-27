from PyQt4.QtCore import QAbstractItemModel, QModelIndex, Qt, QVariant
from PyQt4.QtGui import QColor
from ert.enkf import ErtImplType
from ert_gui.widgets import util


class DataTypeKeysListModel(QAbstractItemModel):
    DEFAULT_DATA_TYPE = QColor(255, 255, 255)
    HAS_OBSERVATIONS = QColor(237, 218, 116)
    GROUP_ITEM = QColor(64, 64, 64)

    def __init__(self, ert):
        """
        @type ert: ert.enkf.EnKFMain
        """
        QAbstractItemModel.__init__(self)
        self.__ert = ert
        self.__keys = self.getAllKeys()
        self.__icon = util.resourceIcon("ide/small/bullet_star")


    def getAllKeys(self):
        """ :rtype: dict of (Str, list) """
        ensemble_config = self.__ert.ensembleConfig()
        keys = {
            "summary": sorted([key for key in ensemble_config.getKeylistFromImplType(ErtImplType.SUMMARY)])
        }

        keys["summary_observation"] = [key for key in keys["summary"] if len(ensemble_config.getNode(key).getObservationKeys()) > 0]

        keys["observation"] = keys["summary_observation"]
        keys["all"] = keys["summary"]

        return keys

    def index(self, row, column, parent=None, *args, **kwargs):
        return self.createIndex(row, column, parent)

    def parent(self, index=None):
        return QModelIndex()

    def rowCount(self, parent=None, *args, **kwargs):
        return len(self.__keys["all"])

    def columnCount(self, QModelIndex_parent=None, *args, **kwargs):
        return 1

    def data(self, index, role=None):
        assert isinstance(index, QModelIndex)

        if index.isValid():
            items = self.__keys["all"]
            row = index.row()
            item = items[row]

            if role == Qt.DisplayRole:
                return item
            elif role == Qt.BackgroundRole:
                if self.isObservationKey(item):
                    return self.HAS_OBSERVATIONS

        return QVariant()

    def itemAt(self, index):
        assert isinstance(index, QModelIndex)

        if index.isValid():
            row = index.row()
            return self.__keys["all"][row]

        return None

    def isObservationKey(self, key):
        return key in self.__keys["observation"]

    def isSummaryKey(self, key):
        return key in self.__keys["summary"]

    def isBlockKey(self, key):
        return False

    def isGenKWKey(self, key):
        return False

    def isGenDataKey(self, key):
        return False

    def isCustomPcaKey(self, key):
        return False






