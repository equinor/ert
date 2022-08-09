from collections import defaultdict
from typing import Dict, List, Optional, Union

from qtpy.QtCore import QAbstractItemModel, QModelIndex, QSize, Qt, QVariant
from qtpy.QtGui import QColor, QFont

from ert.gui.model.snapshot import IsEnsembleRole, ProgressRole, StatusRole


class ProgressProxyModel(QAbstractItemModel):
    def __init__(
        self, source_model: QAbstractItemModel, parent: QModelIndex = None
    ) -> None:
        QAbstractItemModel.__init__(self, parent)
        self._source_model: QAbstractItemModel = source_model
        self._progress: Optional[Dict[Union[str, dict], int]] = None
        self._connect()

    def _connect(self):
        self._source_model.dataChanged.connect(self._source_data_changed)
        self._source_model.rowsInserted.connect(self._source_rows_inserted)
        self._source_model.modelAboutToBeReset.connect(self.modelAboutToBeReset)
        self._source_model.modelReset.connect(self._source_reset)

        # rowCount-1 of the top index in the underlying, will be the last/most
        # recent iteration. If it's -1, then there are no iterations yet.
        last_iter: int = self._source_model.rowCount(QModelIndex()) - 1
        if last_iter >= 0:
            self._recalculate_progress(last_iter)

    # pylint: disable=invalid-name,no-self-use
    def columnCount(self, parent: QModelIndex = None) -> int:
        if parent is None:
            parent = QModelIndex()
        if parent.isValid():
            return 0
        return 1

    # pylint: disable=invalid-name,no-self-use
    def rowCount(self, parent: QModelIndex = None) -> int:
        if parent is None:
            parent = QModelIndex()
        if parent.isValid():
            return 0
        return 1

    def index(self, row: int, column: int, parent: QModelIndex = None) -> QModelIndex:
        if parent is None:
            parent = QModelIndex()
        if parent.isValid():
            return QModelIndex()
        return self.createIndex(row, column, None)

    def parent(self, _index: QModelIndex) -> QModelIndex:
        return QModelIndex()

    # pylint: disable=invalid-name,no-self-use
    def hasChildren(self, parent: QModelIndex) -> bool:
        return not parent.isValid()

    # pylint: disable=too-many-return-statements
    def data(self, index: QModelIndex, role=Qt.DisplayRole) -> QVariant:
        if not index.isValid():
            return QVariant()

        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter

        if role == ProgressRole:
            return self._progress

        if role in (Qt.StatusTipRole, Qt.WhatsThisRole, Qt.ToolTipRole):
            return ""

        if role == Qt.SizeHintRole:
            return QSize(30, 30)

        if role == Qt.FontRole:
            return QFont()

        if role in (Qt.BackgroundRole, Qt.ForegroundRole, Qt.DecorationRole):
            return QColor()

        if role == Qt.DisplayRole:
            return ""

        return QVariant()

    def _recalculate_progress(self, iter_: int):
        status_counts = defaultdict(int)
        nr_reals: int = 0
        current_iter_index = self._source_model.index(iter_, 0, QModelIndex())
        if current_iter_index.internalPointer() is None:
            self._progress = None
            return
        for row in range(0, self._source_model.rowCount(current_iter_index)):
            real_index = self._source_model.index(row, 0, current_iter_index)
            status = real_index.data(StatusRole)
            nr_reals += 1
            status_counts[status] += 1
        self._progress = {"status": status_counts, "nr_reals": nr_reals}

    def _source_data_changed(
        self,
        top_left: QModelIndex,
        _bottom_right: QModelIndex,
        _roles: List[int],
    ):
        if top_left.internalPointer() is None:
            return
        if not top_left.data(IsEnsembleRole):
            return
        self._recalculate_progress(top_left.row())
        index = self.index(0, 0, QModelIndex())
        self.dataChanged.emit(index, index, [ProgressRole])

    def _source_rows_inserted(self, _parent: QModelIndex, start: int, _end: int):
        self._recalculate_progress(start)
        index = self.index(0, 0, QModelIndex())
        self.dataChanged.emit(index, index, [ProgressRole])

    def _source_reset(self):
        self._recalculate_progress(0)
        self.modelReset.emit()
