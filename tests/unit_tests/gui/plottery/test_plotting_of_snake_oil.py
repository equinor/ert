from unittest.mock import Mock

from ert.gui.main import GUILogHandler, _setup_main_window
from ert.gui.tools.plot.data_type_keys_widget import DataTypeKeysWidget
from ert.gui.tools.plot.plot_window import PlotWindow
from ert.services import StorageService


def test_that_all_snake_oil_visualisations_can_be_shown(
    qtbot, snake_oil_case_storage, storage
):
    args_mock = Mock()
    args_mock.config = "snake_oil.ert"

    with StorageService.init_service(
        ert_config="snake_oil.ert",
        project=storage.path,
    ):
        gui = _setup_main_window(snake_oil_case_storage, args_mock, GUILogHandler())
        gui.notifier.set_storage(storage)
        qtbot.addWidget(gui)

        plot_tool = gui.tools["Create plot"]
        plot_tool.trigger()

        qtbot.waitUntil(lambda: gui.findChild(PlotWindow) is not None)
        plot_window = gui.findChild(PlotWindow)
        central_tab = plot_window._central_tab

        # Cycle through showing all the tabs for all keys
        data_types = plot_window.findChild(DataTypeKeysWidget)
        key_list = data_types.data_type_keys_widget
        for i in range(key_list.model().rowCount()):
            key_list.setCurrentIndex(key_list.model().index(i, 0))
            selected_key = data_types.getSelectedItem()
            for i, tab in enumerate(plot_window._plot_widgets):
                if central_tab.isTabEnabled(i):
                    central_tab.setCurrentWidget(tab)
                    assert selected_key["dimensionality"] == tab._plotter.dimensionality
                else:
                    assert selected_key["dimensionality"] != tab._plotter.dimensionality
