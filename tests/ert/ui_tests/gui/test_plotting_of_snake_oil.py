from unittest.mock import Mock

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox, QToolButton

from ert.gui.main import GUILogHandler, _setup_main_window
from ert.gui.tools.plot.data_type_keys_widget import DataTypeKeysWidget
from ert.gui.tools.plot.plot_ensemble_selection_widget import (
    EnsembleSelectListWidget,
)
from ert.gui.tools.plot.plot_window import (
    CROSS_ENSEMBLE_STATISTICS,
    DISTRIBUTION,
    ENSEMBLE,
    GAUSSIAN_KDE,
    HISTOGRAM,
    STATISTICS,
    STD_DEV,
    PlotWindow,
)
from ert.services import ErtServer
from ert.storage import open_storage

from .conftest import get_child, wait_for_child


# Use a fixture for the figure in order for the lifetime
# of the c++ gui element to not go out before mpl_image_compare.
# Note that the data is copied from test-data and all the existing storages
# there will be copied too! They need to be removed!
# Once the storage is created it its cached in .pytest_cache.
@pytest.fixture(
    params=[
        ("FOPR", STATISTICS, "snake_oil"),
        ("FOPR", ENSEMBLE, "snake_oil"),
        ("SNAKE_OIL_PARAM:OP1_OCTAVES", CROSS_ENSEMBLE_STATISTICS, "snake_oil"),
        ("COND", STD_DEV, "heat_equation"),
        ("SNAKE_OIL_PARAM:OP1_OCTAVES", DISTRIBUTION, "snake_oil"),
        ("SNAKE_OIL_PARAM:OP1_OCTAVES", GAUSSIAN_KDE, "snake_oil"),
        ("SNAKE_OIL_PARAM:OP1_OCTAVES", HISTOGRAM, "snake_oil"),
        ("SNAKE_OIL_WPR_DIFF@199", ENSEMBLE, "snake_oil"),
    ],
)
def plot_figure(
    qtbot,
    symlinked_heat_equation_storage_esmda,
    symlinked_snake_oil_case_storage,
    request,
):
    key, plot_name, storage_type = request.param
    args_mock = Mock()

    if storage_type == "snake_oil":
        storage_config = symlinked_snake_oil_case_storage
        args_mock.config = "snake_oil.ert"
    else:
        storage_config = symlinked_heat_equation_storage_esmda
        args_mock.config = "config.ert"

    # For dark storage not to hang
    open_storage(storage_config.ens_path, mode="r")
    log_handler = GUILogHandler()
    with (
        ErtServer.init_service(
            project=storage_config.ens_path,
        ),
    ):
        gui = _setup_main_window(
            storage_config, args_mock, log_handler, storage_config.ens_path
        )
        qtbot.addWidget(gui)

        button_plot_tool = gui.findChild(QToolButton, "button_Create_plot")
        assert button_plot_tool
        qtbot.mouseClick(button_plot_tool, Qt.MouseButton.LeftButton)
        plot_window = wait_for_child(gui, qtbot, PlotWindow)
        central_tab = plot_window._central_tab

        data_types = plot_window.findChild(DataTypeKeysWidget)
        key_list = data_types.data_type_keys_widget
        key_model = key_list.model()
        assert key_model is not None

        case_selection = get_child(
            plot_window, EnsembleSelectListWidget, "ensemble_selector"
        )
        # select all ensembles
        for index in range(case_selection.count()):
            assert (item := case_selection.item(index))
            if not item.data(Qt.ItemDataRole.CheckStateRole):
                case_selection.slot_toggle_plot(item)

        found_selected_key = False
        for i in range(key_model.rowCount()):
            to_select = data_types.model.itemAt(data_types.model.index(i, 0))
            assert to_select is not None
            if to_select.key == key:
                index = key_model.index(i, 0)
                key_list.setCurrentIndex(index)
                selected_key = to_select
                for i, tab in enumerate(plot_window._plot_widgets):
                    if tab.name == plot_name:
                        found_selected_key = True
                        if central_tab.isTabEnabled(i):
                            central_tab.setCurrentWidget(tab)
                            assert (
                                selected_key.dimensionality
                                == tab._plotter.dimensionality
                            )
                            if plot_name == STD_DEV:
                                # we need a better resolution for box plots
                                tab._figure.set_size_inches(
                                    2000 / tab._figure.get_dpi(),
                                    1000 / tab._figure.get_dpi(),
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
@pytest.mark.skip_mac_ci  # test is slow
@pytest.mark.snapshot_test
def test_that_all_snake_oil_visualisations_matches_snapshot(plot_figure):
    return plot_figure


@pytest.mark.skip_mac_ci
def test_that_all_plotter_filter_boxes_yield_expected_filter_results(
    qtbot, snake_oil_case_storage
):
    args_mock = Mock()
    args_mock.config = "snake_oil.ert"

    log_handler = GUILogHandler()
    with (
        ErtServer.init_service(
            project=snake_oil_case_storage.ens_path,
        ),
    ):
        gui = _setup_main_window(
            snake_oil_case_storage,
            args_mock,
            log_handler,
            snake_oil_case_storage.ens_path,
        )
        qtbot.addWidget(gui)

        button_plot_tool = gui.findChild(QToolButton, "button_Create_plot")
        assert button_plot_tool
        qtbot.mouseClick(button_plot_tool, Qt.MouseButton.LeftButton)
        plot_window = wait_for_child(gui, qtbot, PlotWindow)

        key_list = plot_window.findChild(DataTypeKeysWidget).data_type_keys_widget
        item_count = [3, 10, 45]

        assert key_list.model().rowCount() == sum(item_count)
        cbs = plot_window.findChildren(QCheckBox, "FilterCheckBox")

        for i in range(len(item_count)):
            for u, cb in enumerate(cbs):
                cb.setChecked(i == u)

            assert key_list.model().rowCount() in item_count

        plot_window.close()
