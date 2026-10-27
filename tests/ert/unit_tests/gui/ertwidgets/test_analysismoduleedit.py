from collections.abc import Callable

import pytest
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
)
from pytestqt.qtbot import QtBot

from ert.config import ESSettings, LocalizationType, ParameterType
from ert.gui.ertwidgets.analysismoduleedit import AnalysisModuleEdit
from ert.gui.ertwidgets.analysismodulevariablespanel import AnalysisModuleVariablesPanel
from tests.ert.unit_tests.gui.conftest import click_dialog_button


def _click_edit_button_and_interact_with_dialog(
    qtbot: QtBot,
    widget: AnalysisModuleEdit,
    interact_with_dialog: Callable[[QDialog], None],
) -> None:
    def interact() -> None:
        dialog = QApplication.activeModalWidget()
        assert isinstance(dialog, QDialog)
        interact_with_dialog(dialog)

    QTimer.singleShot(0, interact)
    edit_button = widget.findChild(QPushButton, "analysis_variables_popup_button")
    assert edit_button is not None
    qtbot.mouseClick(edit_button, Qt.MouseButton.LeftButton)


def _update_strategy_selector(
    dialog: QDialog, parameter_type: ParameterType
) -> QComboBox:
    panel = dialog.findChild(AnalysisModuleVariablesPanel)
    assert panel is not None
    layout = panel.layout()
    assert isinstance(layout, QFormLayout)
    for combobox in panel.findChildren(QComboBox):
        label = layout.labelForField(combobox)
        if isinstance(label, QLabel) and label.text() == parameter_type.name:
            return combobox
    raise AssertionError(f"No update strategy selector for {parameter_type}")


def _set_update_strategy(
    dialog: QDialog, parameter_type: ParameterType, strategy: LocalizationType
) -> None:
    combobox = _update_strategy_selector(dialog, parameter_type)
    index = combobox.findData(strategy, Qt.ItemDataRole.UserRole)
    assert index != -1
    combobox.setCurrentIndex(index)


def _set_spinbox_value(dialog: QDialog, object_name: str, value: float) -> None:
    spinbox = dialog.findChild(QDoubleSpinBox, object_name)
    assert spinbox is not None
    spinbox.setValue(value)


def _set_threshold_and_truncation_in_dialog(
    dialog: QDialog, correlation_threshold: float, enkf_truncation: float
) -> None:
    _set_spinbox_value(
        dialog, "localization_correlation_threshold", correlation_threshold
    )
    _set_spinbox_value(dialog, "enkf_truncation", enkf_truncation)


def _create_analysis_module_edit(
    qtbot: QtBot,
    es_settings: ESSettings,
    update_strategies: dict[ParameterType, LocalizationType],
    *,
    can_edit_update_strategies: bool = True,
) -> tuple[AnalysisModuleEdit, list[dict[ParameterType, LocalizationType]]]:
    widget = AnalysisModuleEdit(
        es_settings=es_settings,
        get_update_strategies=lambda: update_strategies,
        ensemble_size=10,
        can_edit_update_strategies=lambda: can_edit_update_strategies,
    )
    qtbot.addWidget(widget)
    emitted: list[dict[ParameterType, LocalizationType]] = []
    widget.update_strategies_changed.connect(emitted.append)
    return widget, emitted


@pytest.fixture
def es_settings() -> ESSettings:
    settings = ESSettings()
    settings.localization_correlation_threshold = 0.5
    settings.enkf_truncation = 0.2
    return settings


def test_that_saving_dialog_updates_es_settings_and_emits_edited_update_strategies(
    qtbot: QtBot, es_settings: ESSettings
):
    widget, emitted = _create_analysis_module_edit(
        qtbot, es_settings, {ParameterType.GEN_KW: LocalizationType.GLOBAL}
    )

    def edit_and_save(dialog: QDialog) -> None:
        _set_threshold_and_truncation_in_dialog(dialog, 0.7, 0.3)
        _set_update_strategy(dialog, ParameterType.GEN_KW, LocalizationType.ADAPTIVE)
        click_dialog_button(qtbot, dialog, QDialogButtonBox.StandardButton.Save)

    _click_edit_button_and_interact_with_dialog(qtbot, widget, edit_and_save)

    assert pytest.approx(es_settings.localization_correlation_threshold) == 0.7
    assert pytest.approx(es_settings.enkf_truncation) == 0.3
    assert emitted == [{ParameterType.GEN_KW: LocalizationType.ADAPTIVE}]


def test_that_cancelling_dialog_leaves_es_settings_unchanged_and_emits_nothing(
    qtbot: QtBot, es_settings: ESSettings
):
    widget, emitted = _create_analysis_module_edit(
        qtbot, es_settings, {ParameterType.GEN_KW: LocalizationType.GLOBAL}
    )

    def edit_and_cancel(dialog: QDialog) -> None:
        _set_threshold_and_truncation_in_dialog(dialog, 0.7, 0.3)
        _set_update_strategy(dialog, ParameterType.GEN_KW, LocalizationType.ADAPTIVE)
        click_dialog_button(qtbot, dialog, QDialogButtonBox.StandardButton.Cancel)

    _click_edit_button_and_interact_with_dialog(qtbot, widget, edit_and_cancel)

    assert pytest.approx(es_settings.localization_correlation_threshold) == 0.5
    assert pytest.approx(es_settings.enkf_truncation) == 0.2
    assert emitted == []


def test_that_update_strategy_selectors_are_disabled_when_strategies_are_not_editable(
    qtbot: QtBot, es_settings: ESSettings
):
    widget, _ = _create_analysis_module_edit(
        qtbot,
        es_settings,
        {ParameterType.GEN_KW: LocalizationType.GLOBAL},
        can_edit_update_strategies=False,
    )
    selectors_enabled: list[bool] = []

    def inspect_and_cancel(dialog: QDialog) -> None:
        selectors_enabled.extend(c.isEnabled() for c in dialog.findChildren(QComboBox))
        click_dialog_button(qtbot, dialog, QDialogButtonBox.StandardButton.Cancel)

    _click_edit_button_and_interact_with_dialog(qtbot, widget, inspect_and_cancel)

    assert selectors_enabled
    assert not any(selectors_enabled)


def test_that_saving_dialog_with_non_editable_strategies_saves_es_settings_only(
    qtbot: QtBot, es_settings: ESSettings
):
    widget, emitted = _create_analysis_module_edit(
        qtbot,
        es_settings,
        {ParameterType.GEN_KW: LocalizationType.GLOBAL},
        can_edit_update_strategies=False,
    )

    def edit_and_save(dialog: QDialog) -> None:
        _set_threshold_and_truncation_in_dialog(dialog, 0.7, 0.3)
        click_dialog_button(qtbot, dialog, QDialogButtonBox.StandardButton.Save)

    _click_edit_button_and_interact_with_dialog(qtbot, widget, edit_and_save)

    assert pytest.approx(es_settings.localization_correlation_threshold) == 0.7
    assert pytest.approx(es_settings.enkf_truncation) == 0.3
    assert emitted == []


def test_that_dialog_reads_update_strategies_when_opened_not_when_created(
    qtbot: QtBot, es_settings: ESSettings
):
    current_strategies = {ParameterType.GEN_KW: LocalizationType.GLOBAL}
    widget, _ = _create_analysis_module_edit(qtbot, es_settings, current_strategies)
    current_strategies[ParameterType.GEN_KW] = LocalizationType.ADAPTIVE
    shown_strategies: list[LocalizationType] = []

    def inspect_and_cancel(dialog: QDialog) -> None:
        selector = _update_strategy_selector(dialog, ParameterType.GEN_KW)
        shown_strategies.append(selector.currentData(Qt.ItemDataRole.UserRole))
        click_dialog_button(qtbot, dialog, QDialogButtonBox.StandardButton.Cancel)

    _click_edit_button_and_interact_with_dialog(qtbot, widget, inspect_and_cancel)

    assert shown_strategies == [LocalizationType.ADAPTIVE]


def test_that_saving_dialog_does_not_mutate_provided_update_strategies(
    qtbot: QtBot, es_settings: ESSettings
):
    provided = {ParameterType.GEN_KW: LocalizationType.GLOBAL}
    widget, _ = _create_analysis_module_edit(qtbot, es_settings, provided)

    def edit_and_save(dialog: QDialog) -> None:
        _set_update_strategy(dialog, ParameterType.GEN_KW, LocalizationType.ADAPTIVE)
        click_dialog_button(qtbot, dialog, QDialogButtonBox.StandardButton.Save)

    _click_edit_button_and_interact_with_dialog(qtbot, widget, edit_and_save)

    assert provided == {ParameterType.GEN_KW: LocalizationType.GLOBAL}
