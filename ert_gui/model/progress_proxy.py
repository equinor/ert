from qtpy.QtCore import QRect, QSize, QModelIndex, Qt, QIdentityProxyModel, QVariant

from ert_gui.model.snapshot import SimpleProgressRole, ProgressRole


class ProgressProxyModel(QIdentityProxyModel):
    def __init__(self, parent=None) -> None:
        QIdentityProxyModel.__init__(self, parent)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 1

    def rowCount(self, parent=QModelIndex()) -> int:
        return 1

    def data(self, index: QModelIndex, role=Qt.DisplayRole) -> QVariant:

        # if role == ProgressRole:
        #     d = {}
        #     nr_reals = 0
        #     for k, v in self.sourceModel().root.children.items():
        #         ## iterations
        #         for k1, v1 in v.children.items():
        #             ## realizations
        #             nr_reals += 1
        #             status = v1.data["status"]
        #             if status in d:
        #                 d[status] += 1
        #             else:
        #                 d[status] = 1
        #     return {"status": d, "nr_reals": nr_reals}

        if role == ProgressRole:
            d = {}
            nr_reals = 0
            _, last_item = list(self.sourceModel().root.children.items())[-1]
            print("last ", last_item)
            for _, v in last_item.children.items():
                ## realizations
                nr_reals += 1
                status = v.data["status"]
                if status in d:
                    d[status] += 1
                else:
                    d[status] = 1
            ff = {"status": d, "nr_reals": nr_reals}
            print(ff)
            return ff

        if role == SimpleProgressRole:
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
