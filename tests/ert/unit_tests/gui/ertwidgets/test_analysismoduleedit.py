import pytest
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication, QComboBox, QDialog, QMessageBox, QPushButton
from pytestqt.qtbot import QtBot

from ert.config import ESSettings, GenKwConfig, LocalizationType
from ert.config.parameter_config import supported_localization_types
from ert.gui.ertwidgets.analysismoduleedit import AnalysisModuleEdit
from ert.gui.ertwidgets.analysismodulevariablespanel import AnalysisModuleVariablesPanel


def test_that_localization_choices_match_parameter_type_capabilities(qtbot: QtBot):
    parameter_types = ["gen_kw", "field", "surface"]
    panel = AnalysisModuleVariablesPanel(
        update_strategies={
            parameter_type.upper(): LocalizationType.GLOBAL
            for parameter_type in parameter_types
        },
        correlation_threshold=0.5,
        enkf_truncation=0.9,
    )
    qtbot.addWidget(panel)
    for parameter_type, combobox in zip(
        parameter_types, panel.findChildren(QComboBox), strict=True
    ):
        assert {
            combobox.itemData(index, Qt.ItemDataRole.UserRole)
            for index in range(combobox.count())
        } == supported_localization_types(parameter_type)


@pytest.mark.timeout(10)
def test_that_saving_mixed_strategies_without_changes_preserves_each_parameter(qtbot):
    parameters = [
        GenKwConfig(
            name=str(index),
            distribution={"name": "uniform", "min": 0, "max": 1},
            update_strategy=strategy,
        )
        for index, strategy in enumerate(
            [LocalizationType.GLOBAL, LocalizationType.ADAPTIVE, None]
        )
    ]
    widget = AnalysisModuleEdit(ESSettings(), parameters, 3)
    qtbot.addWidget(widget)

    def accept_without_changes():
        dialog = QApplication.activeModalWidget()
        assert isinstance(dialog, QDialog)
        assert dialog.findChildren(QComboBox)[0].currentText() == "Mixed"
        dialog.accept()

    QTimer.singleShot(0, accept_without_changes)
    widget._show_update_settings_dialog()
    assert [p.update_strategy for p in widget.parameter_config] == [
        LocalizationType.GLOBAL,
        LocalizationType.ADAPTIVE,
        None,
    ]


@pytest.mark.timeout(10)
def test_that_a_changed_parameter_source_rejects_pending_dialog_edits(
    qtbot, monkeypatch
):
    parameter = GenKwConfig(
        name="configured", distribution={"name": "uniform", "min": 0, "max": 1}
    )
    widget = AnalysisModuleEdit(ESSettings(), [parameter], 3)
    qtbot.addWidget(widget)
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args))

    def change_source_and_accept():
        dialog = QApplication.activeModalWidget()
        panel = dialog.findChild(AnalysisModuleVariablesPanel)
        panel._update_strategies["GEN_KW"] = LocalizationType.ADAPTIVE
        widget.parameter_state.select_prior("prior", [parameter])
        dialog.accept()

    QTimer.singleShot(0, change_source_and_accept)
    widget._show_update_settings_dialog()
    assert widget.parameter_config[0].update_strategy == LocalizationType.GLOBAL
    assert len(warnings) == 1


def test_that_reset_button_restores_configured_and_prior_drafts(qtbot):
    parameter = GenKwConfig(
        name="configured", distribution={"name": "uniform", "min": 0, "max": 1}
    )
    widget = AnalysisModuleEdit(ESSettings(), [parameter], 3)
    qtbot.addWidget(widget)
    state = widget.parameter_state
    state.apply_strategies_by_type({"gen_kw": LocalizationType.ADAPTIVE})
    state.select_prior("prior", [parameter])
    state.apply_strategies_by_type({"gen_kw": LocalizationType.ADAPTIVE})
    reset = widget.findChild(QPushButton, "reset_parameter_changes")
    assert reset.isEnabled()
    reset.click()
    assert not reset.isEnabled()
    assert state.overrides == {}
    state.use_configured_parameters()
    assert state.overrides == {}


def test_that_saving_settings_updates_the_draft_and_general_settings(qtbot: QtBot):
    es_settings = ESSettings()
    es_settings.localization_correlation_threshold = 0.5
    es_settings.enkf_truncation = 0.2
    ensemble_size = 10
    parameter = GenKwConfig(
        name="name",
        distribution={"name": "uniform", "min": 0, "max": 1},
        update_strategy=LocalizationType.GLOBAL,
    )
    parameter_config = [parameter]

    widget = AnalysisModuleEdit(
        es_settings=es_settings,
        parameter_config=parameter_config,
        ensemble_size=ensemble_size,
    )
    qtbot.addWidget(widget)

    def inspect_and_accept_dialog() -> None:

        dialog = QApplication.activeModalWidget()
        assert dialog is not None
        assert isinstance(dialog, QDialog)

        panel = dialog.findChild(AnalysisModuleVariablesPanel)
        assert panel is not None

        # Update settings in the panel
        panel._correlation_threshold = 0.7
        panel._enkf_truncation = 0.3
        panel._update_strategies["GEN_KW"] = LocalizationType.ADAPTIVE

        dialog.accept()

    QTimer.singleShot(0, inspect_and_accept_dialog)

    button = widget.findChild(QPushButton)
    assert button is not None
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)

    # After the dialog is accepted, check that the settings are updated
    assert pytest.approx(es_settings.localization_correlation_threshold) == 0.7
    assert pytest.approx(es_settings.enkf_truncation) == 0.3
    assert widget.parameter_config[0].update_strategy == LocalizationType.ADAPTIVE
    assert parameter.update_strategy == LocalizationType.GLOBAL


def test_that_only_parameters_with_update_strategy_are_updated(qtbot: QtBot):
    parameter_without_strategy = GenKwConfig(
        name="without_strategy",
        distribution={"name": "uniform", "min": 0, "max": 1},
        update_strategy=None,
    )
    parameter_with_strategy = GenKwConfig(
        name="with_strategy",
        distribution={"name": "uniform", "min": 0, "max": 1},
        update_strategy=LocalizationType.GLOBAL,
    )

    widget = AnalysisModuleEdit(
        es_settings=ESSettings(),
        parameter_config=[parameter_without_strategy, parameter_with_strategy],
        ensemble_size=10,
    )
    qtbot.addWidget(widget)

    def select_adaptive_strategy_and_accept_dialog() -> None:
        dialog = QApplication.activeModalWidget()
        assert isinstance(dialog, QDialog)

        panel = dialog.findChild(AnalysisModuleVariablesPanel)
        assert panel is not None
        panel._update_strategies["GEN_KW"] = LocalizationType.ADAPTIVE
        dialog.accept()

    QTimer.singleShot(0, select_adaptive_strategy_and_accept_dialog)

    button = widget.findChild(QPushButton)
    assert button is not None
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)

    assert parameter_without_strategy.update_strategy is None
    assert parameter_with_strategy.update_strategy == LocalizationType.GLOBAL
    assert [p.update_strategy for p in widget.parameter_config] == [
        None,
        LocalizationType.ADAPTIVE,
    ]


def test_that_settings_are_not_updated_on_cancel(qtbot: QtBot):
    es_settings = ESSettings()
    es_settings.localization_correlation_threshold = 0.5
    es_settings.enkf_truncation = 0.2
    ensemble_size = 10
    parameter = GenKwConfig(
        name="name",
        distribution={"name": "uniform", "min": 0, "max": 1},
        update_strategy=LocalizationType.GLOBAL,
    )
    parameter_config = [parameter]

    widget = AnalysisModuleEdit(
        es_settings=es_settings,
        parameter_config=parameter_config,
        ensemble_size=ensemble_size,
    )
    qtbot.addWidget(widget)

    def inspect_and_reject_dialog() -> None:
        dialog = QApplication.activeModalWidget()
        assert dialog is not None
        assert isinstance(dialog, QDialog)

        panel = dialog.findChild(AnalysisModuleVariablesPanel)
        assert panel is not None

        # Update settings in the panel
        panel._correlation_threshold = 0.7
        panel._enkf_truncation = 0.3
        panel._update_strategies["GEN_KW"] = LocalizationType.ADAPTIVE

        dialog.reject()

    QTimer.singleShot(0, inspect_and_reject_dialog)

    button = widget.findChild(QPushButton)
    assert button is not None
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)

    # After the dialog is rejected, check that the settings are not updated
    assert pytest.approx(es_settings.localization_correlation_threshold) == 0.5
    assert pytest.approx(es_settings.enkf_truncation) == 0.2
    assert widget.parameter_config[0].update_strategy == LocalizationType.GLOBAL
