from PyQt5.QtCore import QSize
from ert_gui.model.real_list import RealListModel
from ert_gui.model.dev_helpers import create_snapshot
import sys

from qtpy.QtCore import QModelIndex, Slot
from qtpy.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileSystemModel,
    QListView,
    QTableView,
    QToolBar,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from ert_gui.model.job_list import JobListProxyModel
from ert_gui.model.node import NodeType, snapshot_to_tree
from ert_gui.model.snapshot import SnapshotModel


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle("My Awesome App")
        frame = QFrame(self)

        model = SnapshotModel(self)
        
        treew = QTreeView(self)
        treew.setModel(model)

        # models
        simple_progress_proxy_model = SimpleProgressProxyModel(self)
        simple_progress_proxy_model.setSourceModel(model)

        progress_proxy_model = ProgressProxyModel(self)
        progress_proxy_model.setSourceModel(model)

        ## views
        simple_progress_view = SimpleProgressView(self)
        simple_progress_view.setModel(simple_progress_proxy_model)

        progress_view = ProgressView(self)
        progress_view.setModel(progress_proxy_model)

        legend_view = LegendView(self)
        legend_view.setModel(progress_proxy_model)

        real_view = RealizationView(self)
        real_view.setModel(model)


        legend_layout = QHBoxLayout()
        self.legends = {}
        for state, color in REAL_STATE_TO_COLOR.items():
            self.legends[state] = Legend("%s (%d/%d)", QColor(*color))
            self.legends[state].updateLegend(state, 0, 0)
            legend_layout.addWidget(self.legends[state])

        legend_widget_container = QWidget()
        legend_widget_container.setLayout(legend_layout)

        layout = QVBoxLayout()
        layout.addWidget(simple_progress_view)
        layout.addWidget(progress_view)
        layout.addWidget(legend_widget_container)
        layout.addWidget(legend_view)
        layout.addWidget(real_view)
        layout.addWidget(treew)
        frame.setLayout(layout)

        self.setCentralWidget(frame)
        # model = QFileSystemModel()
        # model.setRootPath("/")
        model = SnapshotModel(self)
        model._add_snapshot(snapshot_to_tree(create_snapshot(0)))
        model._add_snapshot(snapshot_to_tree(create_snapshot(0)))
        model._add_snapshot(snapshot_to_tree(create_snapshot(1)))

        self._model = model
        treew = QTreeView(self)
        treew.setModel(model)
        treew.clicked.connect(self._select_iter)

        job_table = QTableView(self)
        job_table.setModel(model)
        self._job_table = job_table

        real_model = RealListModel(self, 0)
        real_model.setSourceModel(model)

        real_grid = QListView(self)
        real_grid.setViewMode(QListView.IconMode)
        real_grid.setGridSize(QSize(120, 20))
        real_grid.setModel(real_model)
        real_grid.setVisible(False)
        real_grid.clicked.connect(self._select_real)
        self._real_grid = real_grid

        vbox_widget = QVBoxLayout()
        vbox_widget.addWidget(treew)
        vbox_widget.addWidget(real_grid)
        vbox_widget.addWidget(job_table)

        central_widget = QWidget(self)
        central_widget.setLayout(vbox_widget)

        self.setCentralWidget(central_widget)

        toolbar = QToolBar("mutate model", self)
        toolbar.addAction("Add snapshot", model.add_snapshot)
        toolbar.addAction("Mutate snapshot 0", model.mutate_snapshot)
        toolbar.addAction(
            "Mutate snapshot 0 (with partial)", model.mutate_snapshot_with_partial
        )
        self.addToolBar(toolbar)

        toolbar.addAction("Add a job on real 0", model.add_job)
        toolbar.addAction("Add a job on real 1", model.add_job1)
        self.addToolBar(toolbar)

    @Slot(QModelIndex)
    def _select_iter(self, index):
        node = index.internalPointer()
        if node is None or node.type != NodeType.ITER:
            return
        iter_ = node.row()
        print("select iter", iter_)

        self._real_grid.model().setIter(iter_)
        self._real_grid.setVisible(True)

    @Slot(QModelIndex)
    def _select_real(self, index):
        node = index.internalPointer()
        if node is None or node.type != NodeType.REAL:
            return
        step = 0
        stage = 0
        real = node.row()
        iter_ = node.parent.row()
        print("select real", step, stage, real, iter_)

        # create a proxy model
        proxy = JobListProxyModel(self, iter_, real, stage, step)
        proxy.setSourceModel(self._model)
        self._job_table.setModel(proxy)


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()
