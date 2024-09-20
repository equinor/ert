import math

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QCheckBox, QDoubleSpinBox

from ert.config import ESSettings
from ert.gui.ertwidgets.analysismodulevariablespanel import AnalysisModuleVariablesPanel


@pytest.fixture
def panel_with_localization_on(qtbot):
    def func(settings, ensemble_size):
        widget = AnalysisModuleVariablesPanel(settings, ensemble_size)
        qtbot.addWidget(widget)
        check_box = widget.findChild(QCheckBox, name="localization")
        qtbot.mouseClick(check_box, Qt.LeftButton)
        return settings, widget

    yield func


def test_that_turning_on_localization_is_saved(panel_with_localization_on):
    settings = ESSettings()
    assert settings.localization == False
    settings, _ = panel_with_localization_on(settings, 123)
    assert settings.localization == True


@pytest.mark.parametrize(
    "ensemble_size, expected",
    [
        (1, 1),
        (8, 1),
        (9, 3 / math.sqrt(9)),
        (200, 3 / math.sqrt(200)),
    ],
)
def test_default_localization_threshold(
    panel_with_localization_on, ensemble_size, expected
):
    settings = ESSettings()
    _, widget = panel_with_localization_on(settings, ensemble_size)

    spinner = widget.findChild(QDoubleSpinBox, name="localization_threshold")
    assert spinner.value() == pytest.approx(expected)


@pytest.mark.parametrize("set_value", [0.0, 0.2, 0.5, 1.0])
def test_setting_localization_threshold(panel_with_localization_on, set_value):
    settings = ESSettings()
    _, widget = panel_with_localization_on(settings, 123)

    spinner = widget.findChild(QDoubleSpinBox, name="localization_threshold")
    spinner.setValue(set_value)
    assert spinner.value() == settings.localization_correlation_threshold == set_value
