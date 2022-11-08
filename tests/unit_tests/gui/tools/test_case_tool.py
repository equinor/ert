from textwrap import dedent
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QPushButton

from ert._c_wrappers.enkf import EnKFMain, ResConfig
from ert._clib.state_map import State
from ert._clib.update import Parameter
from ert.gui.tools.manage_cases.case_init_configuration import (
    CaseInitializationConfigurationPanel,
)


@pytest.mark.usefixtures("copy_poly_case")
def test_case_tool_init_prior(qtbot):
    ert = EnKFMain(ResConfig("poly.ert"))
    storage = ert.getCurrentFileSystem()
    assert (
        list(storage.getStateMap())
        == [State.UNDEFINED] * ert.getEnsembleSize()
    )
    tool = CaseInitializationConfigurationPanel(ert, MagicMock())
    qtbot.mouseClick(
        tool.findChild(QPushButton, name="initialize_scratch_button"), Qt.LeftButton
    )
    assert (
        list(storage.getStateMap())
        == [State.INITIALIZED] * ert.getEnsembleSize()
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
    with open("config.ert", "w") as fh:
        fh.writelines(config)
    with open("template.txt", "w") as fh:
        fh.writelines("MY_KEYWORD <MY_KEYWORD>")
    with open("prior.txt", "w") as fh:
        fh.writelines("MY_KEYWORD NORMAL 0 1")
    ert = EnKFMain(ResConfig("config.ert"))
    storage_manager = ert.storage_manager

    prior = storage_manager["default"]
    ert.sample_prior(prior, list(range(ert.getEnsembleSize())))
    new_case = storage_manager.add_case("new_case")
    ert.switchFileSystem("new_case")
    tool = CaseInitializationConfigurationPanel(ert, MagicMock())
    assert (
        list(new_case.getStateMap())
        == [State.UNDEFINED] * ert.getEnsembleSize()
    )
    qtbot.mouseClick(
        tool.findChild(QPushButton, name="initialize_existing_button"), Qt.LeftButton
    )
    assert new_case.load_parameter(
        ert.ensembleConfig(),
        [0],
        Parameter("KW_NAME"),
    ).flatten()[0] == pytest.approx(0.3797726974728599)
