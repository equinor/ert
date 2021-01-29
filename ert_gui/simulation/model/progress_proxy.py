from qtpy.QtCore import QRect, QSize, QModelIndex, Qt, QIdentityProxyModel, QVariant


class ProgressProxyModel(QIdentityProxyModel):
    def __init__(self, parent=None) -> None:
        QIdentityProxyModel.__init__(self, parent)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 1

    def rowCount(self, parent=QModelIndex()) -> int:
        return 1

    def data(self, index: QModelIndex, role=Qt.DisplayRole) -> QVariant:
        d = {}
        nr_reals = 0
        for k, v in self.sourceModel().root.children.items():
            ## iterations
            for k1, v1 in v.children.items():
                ## realizations
                nr_reals += 1
                status = v1.data["status"]
                if status in d:
                    d[status] += 1
                else:
                    d[status] = 1
        return {"status": d, "nr_reals": nr_reals}
