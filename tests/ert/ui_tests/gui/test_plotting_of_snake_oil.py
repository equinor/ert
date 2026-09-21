from pathlib import Path
from unittest.mock import Mock

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox, QToolButton

from ert.gui.main import GUILogHandler, _setup_main_window
from ert.gui.plotting.plot_window import (
    PlotWindow,
)
from ert.gui.plotting.utils.plot_maps import (
    CROSS_ENSEMBLE_STATISTICS,
    DISTRIBUTION,
    ENSEMBLE,
    HISTOGRAM,
    STATISTICS,
    STD_DEV,
)
from ert.gui.plotting.widgets import DataTypeKeysWidget, EnsembleSelectListWidget
from ert.services import ErtServerController
from ert.storage import open_storage

from .conftest import get_child, wait_for_child


# The tolerance is chosen by guess, in one bug we observed a
# where locations of observations in the standard deviation plot
# were off, we needed a tolerance of 5 to get tests to fail.
# Note that the data is copied from test-data and all the existing storages
# there will be copied too! They need to be removed!
# Once the storage is created it its cached in .pytest_cache.
@pytest.mark.mpl_image_compare(tolerance=5.0, style="default")
@pytest.mark.skip_mac_ci  # test is slow
@pytest.mark.xdist_group(name="uses_heat_equation_storage")
@pytest.mark.parametrize(
    ("key", "plot_name", "storage_type"),
    [
        pytest.param("FOPR", STATISTICS, "snake_oil", id="FOPT-statistics-snake_oil"),
        pytest.param("FOPR", ENSEMBLE, "snake_oil", id="FOPR-ensemble-snake_oil"),
        pytest.param(
            "SNAKE_OIL_PARAM_OP1:OP1_OCTAVES",
            CROSS_ENSEMBLE_STATISTICS,
            "snake_oil",
            id="OCTAVES-cross-snake_oil",
        ),
        pytest.param("COND", STD_DEV, "heat_equation", id="COND-stddev-heat"),
        pytest.param(
            "SNAKE_OIL_PARAM_OP1:OP1_OCTAVES",
            DISTRIBUTION,
            "snake_oil",
            id="OCTAVES-dist-snake_oil",
        ),
        pytest.param(
            "SNAKE_OIL_PARAM_OP1:OP1_OCTAVES",
            HISTOGRAM,
            "snake_oil",
            id="OCTAVES-histogram-snake_oil",
        ),
        pytest.param(
            "SNAKE_OIL_WPR_DIFF@199",
            ENSEMBLE,
            "snake_oil",
            id="WPRDIFF-ensemble-snake_oil",
        ),
    ],
)
def test_that_plot_images_are_unchanged(
    qtbot,
    symlinked_heat_equation_storage_esmda,
    symlinked_snake_oil_case_storage,
    key,
    plot_name,
    storage_type,
):
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
        ErtServerController.init_service(
            project=Path(storage_config.ens_path).absolute(),
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
        figure = None
        for key_index in range(key_model.rowCount()):
            to_select = data_types.model.itemAt(data_types.model.index(key_index, 0))
            assert to_select is not None
            if to_select.key == key:
                index = key_model.index(key_index, 0)
                key_list.setCurrentIndex(index)
                selected_key = to_select
                for widget_index, tab in enumerate(plot_window._plot_widgets):
                    if tab.name == plot_name:
                        found_selected_key = True
                        if central_tab.isTabEnabled(widget_index):
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
                            figure = tab._figure.figure
                        else:
                            assert (
                                selected_key.dimensionality
                                != tab._plotter.dimensionality
                            )
        assert found_selected_key
    return figure


@pytest.mark.skip_mac_ci
def test_that_all_plotter_filter_boxes_yield_expected_filter_results(
    qtbot, snake_oil_case_storage
):
    args_mock = Mock()
    args_mock.config = "snake_oil.ert"

    log_handler = GUILogHandler()
    with (
        ErtServerController.init_service(
            project=Path(snake_oil_case_storage.ens_path).absolute(),
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
