import math

from PyQt6.QtWidgets import QDoubleSpinBox
from pytestqt.qtbot import QtBot

from ert.gui.ertwidgets.analysismodulevariablespanel import AnalysisModuleVariablesPanel


def _create_panel(
    qtbot: QtBot,
    correlation_threshold: float = 0.5,
    enkf_truncation: float = 0.2,
) -> AnalysisModuleVariablesPanel:
    widget = AnalysisModuleVariablesPanel(
        correlation_threshold=correlation_threshold,
        enkf_truncation=enkf_truncation,
    )
    qtbot.addWidget(widget)
    return widget


def test_that_panel_initializes_spinners_from_given_values(qtbot: QtBot):
    widget = _create_panel(qtbot, correlation_threshold=0.5, enkf_truncation=0.2)

    correlation_spinner = widget.findChild(
        QDoubleSpinBox, name="localization_correlation_threshold"
    )
    truncation_spinner = widget.findChild(QDoubleSpinBox, name="enkf_truncation")
    assert math.isclose(correlation_spinner.value(), 0.5)
    assert math.isclose(truncation_spinner.value(), 0.2)


def test_that_changing_spinners_emits_signals_and_updates_properties(qtbot: QtBot):
    widget = _create_panel(qtbot)
    thresholds: list[float] = []
    truncations: list[float] = []
    widget.correlation_threshold_changed.connect(thresholds.append)
    widget.enkf_truncation_changed.connect(truncations.append)

    widget.threshold_spinner.setValue(0.7)
    widget.truncation_spinner.setValue(0.3)

    assert thresholds == [0.7]
    assert truncations == [0.3]
    assert math.isclose(widget.correlation_threshold, 0.7)
    assert math.isclose(widget.enkf_truncation, 0.3)


def test_that_set_values_updates_spinners_without_emitting_signals(qtbot: QtBot):
    widget = _create_panel(qtbot)
    emitted: list[float] = []
    widget.correlation_threshold_changed.connect(emitted.append)
    widget.enkf_truncation_changed.connect(emitted.append)

    widget.set_values(correlation_threshold=0.9, enkf_truncation=0.4)

    assert emitted == []
    assert math.isclose(widget.threshold_spinner.value(), 0.9)
    assert math.isclose(widget.truncation_spinner.value(), 0.4)
    assert math.isclose(widget.correlation_threshold, 0.9)
    assert math.isclose(widget.enkf_truncation, 0.4)
