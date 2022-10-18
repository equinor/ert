from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QPushButton

from ert._c_wrappers.enkf import EnKFMain, ResConfig
from ert._clib.state_map import RealizationStateEnum
from ert.gui.tools.manage_cases.case_init_configuration import (
    CaseInitializationConfigurationPanel,
)


@pytest.mark.usefixtures("copy_poly_case")
def test_case_tool_init_prior(qtbot):
    ert = EnKFMain(ResConfig("poly.ert"))
    storage = ert.getCurrentFileSystem()
    assert (
        list(storage.getStateMap())
        == [RealizationStateEnum.STATE_UNDEFINED] * ert.getEnsembleSize()
    )
    tool = CaseInitializationConfigurationPanel(ert, MagicMock())
    qtbot.mouseClick(
        tool.findChild(QPushButton, name="initialize_scratch_button"), Qt.LeftButton
    )
    assert (
        list(storage.getStateMap())
        == [RealizationStateEnum.STATE_INITIALIZED] * ert.getEnsembleSize()
    )
