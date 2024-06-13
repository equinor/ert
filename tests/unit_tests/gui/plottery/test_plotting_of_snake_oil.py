import sys
from unittest.mock import Mock

import pytest
from qtpy.QtWidgets import QCheckBox

from ert.enkf_main import EnKFMain
from ert.gui.main import GUILogHandler, _setup_main_window
from ert.gui.tools.plot.data_type_keys_widget import DataTypeKeysWidget
from ert.gui.tools.plot.plot_window import (
    CROSS_ENSEMBLE_STATISTICS,
    DISTRIBUTION,
    ENSEMBLE,
    GAUSSIAN_KDE,
    HISTOGRAM,
    STATISTICS,
    PlotWindow,
)
from ert.services import StorageService
from ert.storage import open_storage


@pytest.fixture
def enkf_main_snake_oil(snake_oil_case_storage):
    yield EnKFMain(snake_oil_case_storage)


# Use a fixture for the fligure in order for the lifetime
# of the c++ gui element to not go out before mpl_image_compare
@pytest.fixture(
    params=[
        ("FOPR", STATISTICS),
        ("FOPR", ENSEMBLE),
        ("SNAKE_OIL_PARAM:OP1_OCTAVES", CROSS_ENSEMBLE_STATISTICS),
        ("SNAKE_OIL_PARAM:OP1_OCTAVES", DISTRIBUTION),
        ("SNAKE_OIL_PARAM:OP1_OCTAVES", GAUSSIAN_KDE),
        ("SNAKE_OIL_PARAM:OP1_OCTAVES", HISTOGRAM),
    ],
)
def plot_figure(qtbot, enkf_main_snake_oil, request):
    key = request.param[0]
    plot_name = request.param[1]
    args_mock = Mock()
    args_mock.config = "snake_oil.ert"

    log_handler = GUILogHandler()
    with StorageService.init_service(
        project=enkf_main_snake_oil.ert_config.ens_path,
    ), open_storage(enkf_main_snake_oil.ert_config.ens_path) as storage:
        gui = _setup_main_window(enkf_main_snake_oil, args_mock, log_handler, storage)
        qtbot.addWidget(gui)

        plot_tool = gui.tools["Create plot"]
        plot_tool.trigger()

        qtbot.waitUntil(lambda: gui.findChild(PlotWindow) is not None)
        plot_window = gui.findChild(PlotWindow)
        central_tab = plot_window._central_tab

        data_types = plot_window.findChild(DataTypeKeysWidget)
        key_list = data_types.data_type_keys_widget
        key_model = key_list.model()
        assert key_model is not None

        found_selected_key = False
        for i in range(key_model.rowCount()):
            key_list.setCurrentIndex(key_model.index(i, 0))
            selected_key = data_types.getSelectedItem()
            assert selected_key is not None
            if selected_key.key == key:
                for i, tab in enumerate(plot_window._plot_widgets):
                    if tab.name == plot_name:
                        found_selected_key = True
                        if central_tab.isTabEnabled(i):
                            central_tab.setCurrentWidget(tab)
                            assert (
                                selected_key.dimensionality
                                == tab._plotter.dimensionality
                            )
                            yield tab._figure.figure
                        else:
                            assert (
                                selected_key.dimensionality
                                != tab._plotter.dimensionality
                            )
        assert found_selected_key
        plot_window.close()


# We had an issue where the mpl_image_compare decorator
# was put on an inner function. That makes any failure not
# report so it has to be on a top level test.
# The tolerance is chosen by guess, in one bug we observed a
# mismatch of 58 which would fail the test by being above 10.0
@pytest.mark.mpl_image_compare(tolerance=10.0)
@pytest.mark.skipif(
    sys.platform.startswith("darwin"), reason="Get different size image on mac"
)
def test_that_all_snake_oil_visualisations_matches_snapshot(plot_figure):
    return plot_figure


def test_that_all_plotter_filter_boxes_yield_expected_filter_results(
    qtbot, enkf_main_snake_oil
):
    args_mock = Mock()
    args_mock.config = "snake_oil.ert"

    log_handler = GUILogHandler()
    with StorageService.init_service(
        project=enkf_main_snake_oil.ert_config.ens_path,
    ), open_storage(enkf_main_snake_oil.ert_config.ens_path) as storage:
        gui = _setup_main_window(enkf_main_snake_oil, args_mock, log_handler, storage)
        gui.notifier.set_storage(storage)
        qtbot.addWidget(gui)

        plot_tool = gui.tools["Create plot"]
        plot_tool.trigger()

        qtbot.waitUntil(lambda: gui.findChild(PlotWindow) is not None)
        plot_window = gui.findChild(PlotWindow)

        key_list = plot_window.findChild(DataTypeKeysWidget).data_type_keys_widget
        item_count = [3, 10, 44]

        assert key_list.model().rowCount() == sum(item_count)
        cbs = plot_window.findChildren(QCheckBox, "FilterCheckBox")

        for i in range(len(item_count)):
            for u, cb in enumerate(cbs):
                cb.setChecked(i == u)

            assert key_list.model().rowCount() in item_count

        plot_window.close()
