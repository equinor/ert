import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QPushButton, QTextEdit

from ert._c_wrappers.enkf import EnKFMain, ErtConfig
from ert._c_wrappers.enkf.enums import RealizationStateEnum
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.tools.manage_cases.case_init_configuration import (
    CaseInitializationConfigurationPanel,
)


@pytest.mark.usefixtures("copy_poly_case")
def test_case_tool_init_prior(qtbot, storage):
    ert = EnKFMain(ErtConfig.from_file("poly.ert"))
    notifier = ErtNotifier(ert.ert_config.config_path)
    notifier.set_storage(storage)
    ensemble = storage.create_experiment(
        parameters=ert.ert_config.ensemble_config.parameter_configuration
    ).create_ensemble(
        ensemble_size=ert.getEnsembleSize(),
        name="prior",
    )
    notifier.set_current_case(ensemble)
    assert (
        ensemble.state_map
        == [RealizationStateEnum.STATE_UNDEFINED] * ert.getEnsembleSize()
    )
    tool = CaseInitializationConfigurationPanel(ert, notifier)
    qtbot.mouseClick(
        tool.findChild(QPushButton, name="initialize_from_scratch_button"),
        Qt.LeftButton,
    )
    assert (
        ensemble.state_map
        == [RealizationStateEnum.STATE_INITIALIZED] * ert.getEnsembleSize()
    )


@pytest.mark.usefixtures("copy_poly_case")
def test_case_tool_init_updates_the_case_info_tab(qtbot, storage):
    ert = EnKFMain(ErtConfig.from_file("poly.ert"))
    notifier = ErtNotifier(ert.ert_config.config_path)
    notifier.set_storage(storage)
    ensemble = storage.create_experiment(
        parameters=ert.ert_config.ensemble_config.parameter_configuration
    ).create_ensemble(ensemble_size=ert.getEnsembleSize(), name="default")
    notifier.set_current_case(ensemble)
    tool = CaseInitializationConfigurationPanel(ert, notifier)
    html_edit = tool.findChild(QTextEdit, name="html_text")

    assert not html_edit.toPlainText()
    # Change to the "case info" tab
    tool.setCurrentIndex(2)
    assert "STATE_UNDEFINED" in html_edit.toPlainText()

    # Change to the "initialize from scratch" tab
    tool.setCurrentIndex(1)
    qtbot.mouseClick(
        tool.findChild(QPushButton, name="initialize_from_scratch_button"),
        Qt.LeftButton,
    )

    # Change to the "case info" tab
    tool.setCurrentIndex(2)
    assert "STATE_INITIALIZED" in html_edit.toPlainText()
