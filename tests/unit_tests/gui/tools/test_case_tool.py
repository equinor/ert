import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QPushButton, QTextEdit

from ert.config import ErtConfig
from ert.enkf_main import EnKFMain
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.tools.manage_cases.case_init_configuration import (
    CaseInitializationConfigurationPanel,
)
from ert.realization_state import RealizationState


@pytest.mark.usefixtures("copy_poly_case")
def test_case_tool_init_prior(qtbot, storage):
    config = ErtConfig.from_file("poly.ert")
    notifier = ErtNotifier(config.config_path)
    notifier.set_storage(storage)
    ensemble = storage.create_experiment(
        parameters=config.ensemble_config.parameter_configuration
    ).create_ensemble(
        ensemble_size=config.model_config.num_realizations,
        name="prior",
    )
    notifier.set_current_case(ensemble)
    assert (
        ensemble.state_map
        == [RealizationState.UNDEFINED] * config.model_config.num_realizations
    )
    tool = CaseInitializationConfigurationPanel(
        config, notifier, config.model_config.num_realizations
    )
    qtbot.mouseClick(
        tool.findChild(QPushButton, name="initialize_from_scratch_button"),
        Qt.LeftButton,
    )
    assert (
        ensemble.state_map
        == [RealizationState.INITIALIZED] * config.model_config.num_realizations
    )


@pytest.mark.usefixtures("copy_poly_case")
def test_case_tool_init_updates_the_case_info_tab(qtbot, storage):
    config = ErtConfig.from_file("poly.ert")
    notifier = ErtNotifier(config.config_path)
    notifier.set_storage(storage)
    ensemble = storage.create_experiment(
        parameters=config.ensemble_config.parameter_configuration
    ).create_ensemble(
        ensemble_size=config.model_config.num_realizations, name="default"
    )
    notifier.set_current_case(ensemble)
    tool = CaseInitializationConfigurationPanel(
        config, notifier, config.model_config.num_realizations
    )
    html_edit = tool.findChild(QTextEdit, name="html_text")

    assert not html_edit.toPlainText()
    # Change to the "case info" tab
    tool.setCurrentIndex(2)
    assert "UNDEFINED" in html_edit.toPlainText()

    # Change to the "initialize from scratch" tab
    tool.setCurrentIndex(1)
    qtbot.mouseClick(
        tool.findChild(QPushButton, name="initialize_from_scratch_button"),
        Qt.LeftButton,
    )

    # Change to the "case info" tab
    tool.setCurrentIndex(2)
    assert "INITIALIZED" in html_edit.toPlainText()
