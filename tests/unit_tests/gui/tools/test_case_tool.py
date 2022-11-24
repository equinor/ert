from textwrap import dedent
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QPushButton

from ert._c_wrappers.enkf import EnKFMain, ErtConfig
from ert._clib.state_map import RealizationStateEnum
from ert.gui.tools.manage_cases.case_init_configuration import (
    CaseInitializationConfigurationPanel,
)


@pytest.mark.usefixtures("copy_poly_case")
def test_case_tool_init_prior(qtbot):
    ert = EnKFMain(ErtConfig.from_file("poly.ert"))
    storage = ert.storage_manager.current_case
    assert (
        list(storage.getStateMap())
        == [RealizationStateEnum.STATE_UNDEFINED] * ert.getEnsembleSize()
    )
    tool = CaseInitializationConfigurationPanel(ert, MagicMock())
    qtbot.mouseClick(
        tool.findChild(QPushButton, name="initialize_from_scratch_button"),
        Qt.LeftButton,
    )
    assert (
        list(storage.getStateMap())
        == [RealizationStateEnum.STATE_INITIALIZED] * ert.getEnsembleSize()
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_case_tool_can_copy_case_state(qtbot):
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
    storage_manager = ert.storage_manager

    prior = storage_manager["default"]
    ert.sample_prior(prior, list(range(ert.getEnsembleSize())))
    new_case = storage_manager.add_case("new_case")
    ert.switchFileSystem("new_case")
    tool = CaseInitializationConfigurationPanel(ert, MagicMock())
    assert (
        list(new_case.getStateMap())
        == [RealizationStateEnum.STATE_UNDEFINED] * ert.getEnsembleSize()
    )
    qtbot.mouseClick(
        tool.findChild(QPushButton, name="initialize_existing_button"), Qt.LeftButton
    )
    kw_data = new_case.load_gen_kw(
        "KW_NAME",
        [0],
    ).flatten()
    assert kw_data[0] == pytest.approx(-0.8814227775506998)
