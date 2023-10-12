from qtpy.QtCore import QModelIndex

from ert.gui.model.progress_proxy import ProgressProxyModel
from ert.gui.model.snapshot import SnapshotModel
from ert.gui.simulation.view.legend import LegendView


def test_delegate_instantiated(qtbot, large_snapshot):
    snapshot_model = SnapshotModel()
    progress_proxy_model = ProgressProxyModel(snapshot_model)
    snapshot_model._add_snapshot(SnapshotModel.prerender(large_snapshot), 0)

    legend_view = LegendView()
    qtbot.addWidget(legend_view)
    legend_view.setModel(progress_proxy_model)
    assert legend_view.itemDelegate(QModelIndex())
