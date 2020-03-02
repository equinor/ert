import shutil
import os
import qtpy

from ert_gui.ertnotifier import ErtNotifier
from ert_gui.tools.plot import PlotWindow
from res.enkf import ResConfig, EnKFMain

from ert_shared import ERT

def test_plot_window(monkeypatch, tmpdir, qtbot, source_root):

    with tmpdir.as_cwd():
        test_data_dir = os.path.join(source_root, "test-data", "local", "snake_oil")
        shutil.copytree(test_data_dir, "test_data")
        os.chdir(os.path.join("test_data"))

        res_config = ResConfig("snake_oil.ert")
        ert = EnKFMain(res_config, strict=True)

        notifier = ErtNotifier(ert, "snake_oil")
        ERT.adapt(notifier)

        plot_window = PlotWindow("snake_oil.ert", None)
        qtbot.addWidget(plot_window)

        data_type= plot_window.findChild(qtpy.QtWidgets.QWidget, name="data_keys_list")
        model = data_type.currentIndex()

        for i in range(10):
            data_type.setCurrentIndex(model.sibling(i, 0))
