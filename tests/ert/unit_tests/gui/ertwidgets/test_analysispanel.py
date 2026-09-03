import math

import pytest
from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox
from pytestqt.qtbot import QtBot

from ert.config import ESSettings, LocalizationType
from ert.gui.ertwidgets.analysismodulevariablespanel import AnalysisModuleVariablesPanel


def test_that_empty_update_strategies_are_set_to_global(qtbot: QtBot):
    settings = ESSettings()
    settings.localization_correlation_threshold = 0.5
    settings.enkf_truncation = 0.2
    update_strategies = {}

    widget = AnalysisModuleVariablesPanel(
        update_strategies=update_strategies,
        correlation_threshold=0.5,
        enkf_truncation=0.2,
    )

    qtbot.addWidget(widget)

    comboboxes = widget.findChildren(QComboBox)
    assert len(comboboxes) == 3

    for combobox in comboboxes:
        if (
            combobox.objectName() == "GEN_KW"
            or combobox.objectName() == "SURFACE"
            or combobox.objectName() == "FIELD"
        ):
            assert combobox.currentData() == LocalizationType.GLOBAL


def test_that_the_panel_initializes_with_correct_values(qtbot: QtBot):
    settings = ESSettings()
    settings.localization_correlation_threshold = 0.5
    settings.enkf_truncation = 0.2
    update_strategies = {
        "GEN_KW": LocalizationType.GLOBAL,
        "SURFACE": LocalizationType.DISTANCE,
        "FIELD": LocalizationType.ADAPTIVE,
    }

    widget = AnalysisModuleVariablesPanel(
        update_strategies=update_strategies,
        correlation_threshold=0.5,
        enkf_truncation=0.2,
    )

    qtbot.addWidget(widget)

    correlation_spinner = widget.findChild(
        QDoubleSpinBox, name="localization_correlation_threshold"
    )
    truncation_spinner = widget.findChild(QDoubleSpinBox, name="enkf_truncation")

    assert math.isclose(
        correlation_spinner.value(), settings.localization_correlation_threshold
    )
    assert math.isclose(truncation_spinner.value(), settings.enkf_truncation)

    comboboxes = widget.findChildren(QComboBox)
    assert len(comboboxes) == 3

    for combobox in comboboxes:
        if combobox.objectName() == "GEN_KW":
            assert combobox.currentData() == LocalizationType.GLOBAL
        elif combobox.objectName() == "SURFACE":
            assert combobox.currentData() == LocalizationType.DISTANCE
        elif combobox.objectName() == "FIELD":
            assert combobox.currentData() == LocalizationType.ADAPTIVE


@pytest.mark.parametrize(
    ("object_name", "property_name", "changed_value"),
    [
        ("localization_correlation_threshold", "correlation_threshold", 0.7),
        ("enkf_truncation", "enkf_truncation", 0.4),
    ],
)
def test_that_changing_numeric_control_updates_corresponding_property(
    qtbot: QtBot,
    object_name: str,
    property_name: str,
    changed_value: float,
) -> None:
    widget = AnalysisModuleVariablesPanel(
        update_strategies={},
        correlation_threshold=0.5,
        enkf_truncation=0.2,
    )
    qtbot.addWidget(widget)

    spinner = widget.findChild(QDoubleSpinBox, name=object_name)

    assert spinner is not None
    spinner.setValue(changed_value)
    assert math.isclose(getattr(widget, property_name), changed_value)


@pytest.mark.parametrize(
    ("parameter_type", "initial_strategy", "changed_strategy"),
    [
        ("GEN_KW", LocalizationType.GLOBAL, LocalizationType.ADAPTIVE),
        ("FIELD", LocalizationType.ADAPTIVE, LocalizationType.DISTANCE),
        ("SURFACE", LocalizationType.DISTANCE, LocalizationType.GLOBAL),
    ],
)
def test_that_changing_localization_control_updates_parameter_strategy_property(
    qtbot: QtBot,
    parameter_type: str,
    initial_strategy: LocalizationType,
    changed_strategy: LocalizationType,
) -> None:
    update_strategies = {
        "GEN_KW": LocalizationType.GLOBAL,
        "FIELD": LocalizationType.ADAPTIVE,
        "SURFACE": LocalizationType.DISTANCE,
    }
    widget = AnalysisModuleVariablesPanel(
        update_strategies=update_strategies,
        correlation_threshold=0.5,
        enkf_truncation=0.2,
    )
    qtbot.addWidget(widget)

    matching_comboboxes = [
        combobox
        for combobox in widget.findChildren(QComboBox)
        if combobox.currentData() == initial_strategy
    ]
    assert len(matching_comboboxes) == 1

    combobox = matching_comboboxes[0]
    changed_index = combobox.findData(changed_strategy)
    assert changed_index != -1
    combobox.setCurrentIndex(changed_index)

    assert widget.update_strategies[parameter_type] == changed_strategy
