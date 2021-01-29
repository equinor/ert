from qtpy.QtCore import QRect, QSize, QModelIndex, Qt, QIdentityProxyModel, QVariant


class SimpleProgressProxyModel(QIdentityProxyModel):
    def __init__(self, parent=None) -> None:
        QIdentityProxyModel.__init__(self, parent)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 1

    def rowCount(self, parent=QModelIndex()) -> int:
        return 1

    def data(self, index: QModelIndex, role=Qt.DisplayRole) -> QVariant:
        total_iterations = 0
        finished = 0
        for k, v in self.sourceModel().root.children.items():
            ## iterations
            for k1, v1 in v.children.items():
                ## realizations
                total_iterations += 1
                if v1.data["status"] == "Finished":
                    finished += 1
        return finished / total_iterations
