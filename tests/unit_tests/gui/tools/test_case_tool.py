from textwrap import dedent
from unittest.mock import MagicMock

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
    notifier = ErtNotifier(ert.res_config.config_path)
    notifier.set_storage(storage)
    ensemble = storage.create_experiment().create_ensemble(
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


@pytest.mark.usefixtures("use_tmpdir")
def test_that_case_tool_can_copy_case_state(qtbot, storage):
    """Test that we can copy state from one case to another, first
    need to set up a prior so we have a case with data, then copy
    that case into a new one.
    """
    config = dedent(
        """
    NUM_REALIZATIONS 1
    GEN_KW KW_NAME template.txt kw.txt prior.txt
    RANDOM_SEED 1234
    """
    )
    with open("config.ert", "w", encoding="utf-8") as fh:
        fh.writelines(config)
    with open("template.txt", "w", encoding="utf-8") as fh:
        fh.writelines("MY_KEYWORD <MY_KEYWORD>")
    with open("prior.txt", "w", encoding="utf-8") as fh:
        fh.writelines("MY_KEYWORD NORMAL 0 1")
    ert = EnKFMain(ErtConfig.from_file("config.ert"))
    notifier = ErtNotifier(ert.res_config.config_path)
    notifier.set_storage(storage)

    experiment_id = storage.create_experiment()
    prior = storage.create_ensemble(
        experiment_id, name="prior", ensemble_size=ert.getEnsembleSize()
    )
    ert.sample_prior(prior, list(range(ert.getEnsembleSize())))
    new_case = storage.create_ensemble(
        experiment_id, name="new_case", ensemble_size=ert.getEnsembleSize()
    )
    notifier.set_current_case(new_case)
    tool = CaseInitializationConfigurationPanel(ert, notifier)
    assert (
        new_case.state_map
        == [RealizationStateEnum.STATE_UNDEFINED] * ert.getEnsembleSize()
    )
    qtbot.mouseClick(
        tool.findChild(QPushButton, name="initialize_existing_button"), Qt.LeftButton
    )
    kw_data = new_case.load_gen_kw("KW_NAME", [0]).flatten()
    assert kw_data[0] == pytest.approx(-0.8814227775506998)


@pytest.mark.usefixtures("copy_poly_case")
def test_case_tool_init_updates_the_case_info_tab(qtbot):
    ert = EnKFMain(ErtConfig.from_file("poly.ert"))
    tool = CaseInitializationConfigurationPanel(ert, MagicMock())
    html_edit = tool.findChild(QTextEdit, name="html_text")

    assert not html_edit.toPlainText()
    # Change to the "case info" tab
    tool.setCurrentIndex(3)
    assert "STATE_UNDEFINED" in html_edit.toPlainText()

    # Change to the "initialize from scratch" tab
    tool.setCurrentIndex(1)
    qtbot.mouseClick(
        tool.findChild(QPushButton, name="initialize_from_scratch_button"),
        Qt.LeftButton,
    )

    # Change to the "case info" tab
    tool.setCurrentIndex(3)
    assert "STATE_INITIALIZED" in html_edit.toPlainText()
