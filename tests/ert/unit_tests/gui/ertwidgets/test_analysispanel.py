import math

import pytest
from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox
from pytestqt.qtbot import QtBot

from ert.config import LocalizationType, ParameterType
from ert.gui.ertwidgets.analysismodulevariablespanel import AnalysisModuleVariablesPanel

_MIXED_UPDATE_STRATEGIES = {
    ParameterType.GEN_KW: LocalizationType.GLOBAL,
    ParameterType.FIELD: LocalizationType.ADAPTIVE,
    ParameterType.SURFACE: LocalizationType.DISTANCE,
}


def _create_panel(
    qtbot: QtBot,
    update_strategies: dict[ParameterType, LocalizationType],
    correlation_threshold: float = 0.5,
    enkf_truncation: float = 0.2,
) -> AnalysisModuleVariablesPanel:
    widget = AnalysisModuleVariablesPanel(
        update_strategies=update_strategies,
        correlation_threshold=correlation_threshold,
        enkf_truncation=enkf_truncation,
    )
    qtbot.addWidget(widget)
    return widget


def _strategy_combobox(
    widget: AnalysisModuleVariablesPanel, parameter_type: ParameterType
) -> QComboBox:
    combobox = widget.findChild(QComboBox, name=parameter_type)
    assert combobox is not None, f"No strategy combobox for {parameter_type}"
    return combobox


def test_that_panel_only_has_strategy_comboboxes_for_given_parameter_types(
    qtbot: QtBot,
):
    widget = _create_panel(qtbot, {ParameterType.FIELD: LocalizationType.GLOBAL})

    assert [c.objectName() for c in widget.findChildren(QComboBox)] == [
        ParameterType.FIELD
    ]


@pytest.mark.parametrize("parameter_type", list(ParameterType))
def test_that_strategy_combobox_only_offers_supported_localization_types(
    qtbot: QtBot, parameter_type: ParameterType
):
    widget = _create_panel(qtbot, {parameter_type: LocalizationType.GLOBAL})

    combobox = _strategy_combobox(widget, parameter_type)
    offered = {combobox.itemData(index) for index in range(combobox.count())}

    assert offered == set(parameter_type.supported_localization_types)


def test_that_panel_initializes_controls_from_given_strategies_and_settings(
    qtbot: QtBot,
):
    widget = _create_panel(
        qtbot,
        dict(_MIXED_UPDATE_STRATEGIES),
        correlation_threshold=0.5,
        enkf_truncation=0.2,
    )

    correlation_spinner = widget.findChild(
        QDoubleSpinBox, name="localization_correlation_threshold"
    )
    truncation_spinner = widget.findChild(QDoubleSpinBox, name="enkf_truncation")
    assert math.isclose(correlation_spinner.value(), 0.5)
    assert math.isclose(truncation_spinner.value(), 0.2)

    for parameter_type, strategy in _MIXED_UPDATE_STRATEGIES.items():
        assert _strategy_combobox(widget, parameter_type).currentData() == strategy


@pytest.mark.parametrize(
    ("parameter_type", "changed_strategy"),
    [
        (ParameterType.GEN_KW, LocalizationType.ADAPTIVE),
        (ParameterType.FIELD, LocalizationType.DISTANCE),
        (ParameterType.SURFACE, LocalizationType.GLOBAL),
    ],
)
def test_that_changing_localization_control_updates_parameter_strategy_property(
    qtbot: QtBot,
    parameter_type: ParameterType,
    changed_strategy: LocalizationType,
):
    widget = _create_panel(qtbot, dict(_MIXED_UPDATE_STRATEGIES))

    combobox = _strategy_combobox(widget, parameter_type)
    changed_index = combobox.findData(changed_strategy)
    assert changed_index != -1
    combobox.setCurrentIndex(changed_index)

    assert widget.update_strategies[parameter_type] == changed_strategy
