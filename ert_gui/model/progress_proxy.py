from qtpy.QtCore import QRect, QSize, QModelIndex, Qt, QIdentityProxyModel, QVariant

from ert_gui.model.snapshot import ProgressRole


class ProgressProxyModel(QIdentityProxyModel):
    def __init__(self, parent=None) -> None:
        QIdentityProxyModel.__init__(self, parent)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 1

    def rowCount(self, parent=QModelIndex()) -> int:
        return 1

    def data(self, index: QModelIndex, role=Qt.DisplayRole) -> QVariant:
        if role == ProgressRole:
            d = {}
            nr_reals = 0
            _, last_item = list(self.sourceModel().root.children.items())[-1]
            for _, v in last_item.children.items():
                ## realizations
                nr_reals += 1
                status = v.data["status"]
                if status in d:
                    d[status] += 1
                else:
                    d[status] = 1
            return {"status": d, "nr_reals": nr_reals}
