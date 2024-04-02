import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QPushButton, QTextEdit

from ert.config import ErtConfig
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets.storage_widget import StorageWidget
from ert.gui.tools.manage_experiments.ensemble_init_configuration import (
    EnsembleInitializationConfigurationPanel,
)
from ert.storage.realization_storage_state import RealizationStorageState


@pytest.mark.usefixtures("copy_poly_case")
def test_ensemble_tool_init_prior(qtbot, storage):
    config = ErtConfig.from_file("poly.ert")
    notifier = ErtNotifier(config.config_path)
    notifier.set_storage(storage)
    ensemble = storage.create_experiment(
        parameters=config.ensemble_config.parameter_configuration, name="my-experiment"
    ).create_ensemble(
        ensemble_size=config.model_config.num_realizations,
        name="prior",
    )
    notifier.set_current_ensemble(ensemble)
    assert (
        ensemble.get_ensemble_state()
        == [RealizationStorageState.UNDEFINED] * config.model_config.num_realizations
    )

    tool = EnsembleInitializationConfigurationPanel(
        config, notifier, config.model_config.num_realizations
    )
    qtbot.mouseClick(
        tool.findChild(QPushButton, name="initialize_from_scratch_button"),
        Qt.LeftButton,
    )
    assert (
        ensemble.get_ensemble_state()
        == [RealizationStorageState.INITIALIZED] * config.model_config.num_realizations
    )


@pytest.mark.usefixtures("copy_poly_case")
def test_that_init_updates_the_info_tab(qtbot, storage):
    config = ErtConfig.from_file("poly.ert")
    notifier = ErtNotifier(config.config_path)
    notifier.set_storage(storage)

    ensemble = storage.create_experiment(
        parameters=config.ensemble_config.parameter_configuration, name="my-experiment"
    ).create_ensemble(
        ensemble_size=config.model_config.num_realizations, name="default"
    )
    notifier.set_current_ensemble(ensemble)

    tool = EnsembleInitializationConfigurationPanel(
        config, notifier, config.model_config.num_realizations
    )

    html_edit = tool.findChild(QTextEdit, name="html_text")
    assert not html_edit.toPlainText()

    # select the created ensemble
    storage_widget = tool.findChild(StorageWidget)
    storage_widget._tree_view.expandAll()
    model_index = storage_widget._tree_view.model().index(
        0, 0, storage_widget._tree_view.model().index(0, 0)
    )
    storage_widget._tree_view.setCurrentIndex(model_index)

    assert "UNDEFINED" in html_edit.toPlainText()
    assert not "RealizationStorageState.UNDEFINED" in html_edit.toPlainText()

    # Change to the "initialize from scratch" tab
    tool.setCurrentIndex(1)
    qtbot.mouseClick(
        tool.findChild(QPushButton, name="initialize_from_scratch_button"),
        Qt.LeftButton,
    )

    # Change back to first tab
    tool.setCurrentIndex(0)
    assert "INITIALIZED" in html_edit.toPlainText()
    assert not "RealizationStorageState.INITIALIZED" in html_edit.toPlainText()
