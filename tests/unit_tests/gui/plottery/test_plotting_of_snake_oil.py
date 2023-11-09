from unittest.mock import Mock

import pytest

from ert.enkf_main import EnKFMain
from ert.gui.main import GUILogHandler, _setup_main_window
from ert.gui.tools.plot.data_type_keys_widget import DataTypeKeysWidget
from ert.gui.tools.plot.plot_window import (
    CROSS_CASE_STATISTICS,
    DISTRIBUTION,
    ENSEMBLE,
    GAUSSIAN_KDE,
    HISTOGRAM,
    STATISTICS,
    PlotWindow,
)
from ert.services import StorageService


@pytest.fixture
def enkf_main_snake_oil(snake_oil_case_storage):
    yield EnKFMain(snake_oil_case_storage)


@pytest.mark.parametrize(
    "key, plot_name",
    [
        ("FOPR", STATISTICS),
        ("FOPR", ENSEMBLE),
        ("SNAKE_OIL_PARAM:OP1_OCTAVES", CROSS_CASE_STATISTICS),
        ("SNAKE_OIL_PARAM:OP1_OCTAVES", DISTRIBUTION),
        ("SNAKE_OIL_PARAM:OP1_OCTAVES", GAUSSIAN_KDE),
        ("SNAKE_OIL_PARAM:OP1_OCTAVES", HISTOGRAM),
    ],
)
def test_that_all_snake_oil_visualisations_matches_snapshot(
    qtbot, enkf_main_snake_oil, storage, plot_name, key
):
    args_mock = Mock()
    args_mock.config = "snake_oil.ert"

    with StorageService.init_service(
        ert_config="snake_oil.ert",
        project=storage.path,
    ):
        gui = _setup_main_window(enkf_main_snake_oil, args_mock, GUILogHandler())
        gui.notifier.set_storage(storage)
        qtbot.addWidget(gui)

        plot_tool = gui.tools["Create plot"]
        plot_tool.trigger()

        qtbot.waitUntil(lambda: gui.findChild(PlotWindow) is not None)
        plot_window = gui.findChild(PlotWindow)
        central_tab = plot_window._central_tab

        # Use an inner function in order for the lifetime
        # of the c++ gui element to not go out before mpl_image_compare
        @pytest.mark.mpl_image_compare(tolerance=10)
        def inner():
            # Cycle through showing all the tabs for all keys
            data_types = plot_window.findChild(DataTypeKeysWidget)
            key_list = data_types.data_type_keys_widget
            for i in range(key_list.model().rowCount()):
                key_list.setCurrentIndex(key_list.model().index(i, 0))
                selected_key = data_types.getSelectedItem()
                if selected_key["key"] == key:
                    for i, tab in enumerate(plot_window._plot_widgets):
                        if tab.name == plot_name:
                            if central_tab.isTabEnabled(i):
                                central_tab.setCurrentWidget(tab)
                                assert (
                                    selected_key["dimensionality"]
                                    == tab._plotter.dimensionality
                                )
                                return tab._figure.figure
                            else:
                                assert (
                                    selected_key["dimensionality"]
                                    != tab._plotter.dimensionality
                                )

        inner()
        plot_window.close()
