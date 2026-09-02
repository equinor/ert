import math

import pytest
from PyQt6.QtWidgets import QDoubleSpinBox
from pytestqt.qtbot import QtBot

from ert.config import ESSettings
from ert.gui.ertwidgets.analysismodulevariablespanel import AnalysisModuleVariablesPanel




@pytest.mark.skip("refactor")
@pytest.mark.parametrize("set_value", [0.0, 0.2, 0.5, 1.0])
def test_that_setting_localization_threshold_updates_analysis_settings(
    qtbot: QtBot, set_value
):
    settings = ESSettings()
    widget = AnalysisModuleVariablesPanel(settings, 123)
    qtbot.addWidget(widget)

    spinner = widget.findChild(
        QDoubleSpinBox, name="localization_correlation_threshold"
    )
    spinner.setValue(set_value)
    assert spinner.value() == settings.localization_correlation_threshold == set_value
