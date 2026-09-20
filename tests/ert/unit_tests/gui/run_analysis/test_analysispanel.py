import math

import pytest
from PyQt6.QtWidgets import QDoubleSpinBox
from pytestqt.qtbot import QtBot

from ert.config import ESSettings
from ert.gui.ertwidgets.analysismodulevariablespanel import AnalysisModuleVariablesPanel


@pytest.mark.parametrize(
    ("ensemble_size", "expected"),
    [
        (1, 1),
        (8, 1),
        (9, 3 / math.sqrt(9)),
        (200, 3 / math.sqrt(200)),
    ],
)
def test_that_default_localization_threshold_depends_on_ensemble_size(
    qtbot: QtBot, ensemble_size, expected
):
    settings = ESSettings()
    widget = AnalysisModuleVariablesPanel(settings, ensemble_size)
    qtbot.addWidget(widget)

    spinner = widget.findChild(
        QDoubleSpinBox, name="localization_correlation_threshold"
    )
    assert spinner.value() == pytest.approx(expected)


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
