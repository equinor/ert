from qtpy.QtCore import QModelIndex, Qt, Signal, Slot
from qtpy.QtWidgets import QTableView, QFrame, QVBoxLayout


from ert_gui.model.snapshot import NodeRole, SnapshotModel
from ert_gui.model.job_list import JobListProxyModel
from ert_gui.model.node import NodeType
from ert_gui.model.real_list import RealListModel

from ert_gui.simulation.view.realization import RealizationView


class IterationWidget(QFrame):
    def __init__(self, iter: int, parent=None) -> None:
        super(IterationWidget, self).__init__(parent)

        self._iteration = iter

        self._real_view = RealizationView(self)
        self._real_view.clicked.connect(self._select_real)

        self._job_view = QTableView(self)

        layout = QVBoxLayout()
        layout.addWidget(self._real_view)
        layout.addWidget(self._job_view)

        self.setLayout(layout)

    def setModel(self, model: SnapshotModel) -> None:

        self._real_list_model = RealListModel(self, self._iteration)
        self._real_list_model.setSourceModel(model)

        self._real_view.setModel(self._real_list_model)
        self._real_view.model().setIter(self._iteration)

        self._job_model = JobListProxyModel(self, self._iteration, 0, 0, 0)
        self._job_model.setSourceModel(model)
        self._job_view.setModel(self._job_model)

        # for select_real
        self._snapshot_model = model

    @Slot(QModelIndex)
    def _select_real(self, index) -> None:
        node = index.internalPointer()
        if node is None or node.type != NodeType.REAL:
            return
        step = 0
        stage = 0
        real = node.row()
        iter_ = node.parent.row()
        print("select real", step, stage, real, iter_)

        # create a proxy model
        # TODO: change values on proxymodel such that it does not need creation
        proxy = JobListProxyModel(self, iter_, real, stage, step)
        proxy.setSourceModel(self._snapshot_model)
        self._job_view.setModel(proxy)
