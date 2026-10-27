import pytest
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication, QComboBox, QDialog, QPushButton
from pytestqt.qtbot import QtBot

from ert.config import ESSettings, GenKwConfig, LocalizationType
from ert.gui.ertwidgets.analysismoduleedit import AnalysisModuleEdit
from ert.gui.ertwidgets.analysismodulevariablespanel import AnalysisModuleVariablesPanel


def test_that_settings_are_updated_correctly(qtbot: QtBot):
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
    assert widget._parameter_config[0].update_strategy == LocalizationType.ADAPTIVE


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
    assert parameter_with_strategy.update_strategy == LocalizationType.ADAPTIVE


def test_that_locked_update_strategies_disable_comboboxes_and_are_not_applied(
    qtbot: QtBot,
):
    es_settings = ESSettings()
    parameter = GenKwConfig(
        name="name",
        distribution={"name": "uniform", "min": 0, "max": 1},
        update_strategy=LocalizationType.GLOBAL,
    )
    settings_changed: list[bool] = []
    widget = AnalysisModuleEdit(
        es_settings=es_settings,
        parameter_config=[parameter],
        ensemble_size=10,
        can_edit_update_strategies=lambda: False,
    )
    widget.settings_changed.connect(lambda: settings_changed.append(True))
    qtbot.addWidget(widget)
    comboboxes_enabled: list[bool] = []

    def edit_and_accept_dialog() -> None:
        dialog = QApplication.activeModalWidget()
        assert isinstance(dialog, QDialog)
        panel = dialog.findChild(AnalysisModuleVariablesPanel)
        assert panel is not None
        comboboxes = panel.findChildren(QComboBox)
        comboboxes_enabled.extend(combobox.isEnabled() for combobox in comboboxes)
        panel.truncation_spinner.setValue(0.3)
        comboboxes[0].setCurrentIndex(
            comboboxes[0].findData(LocalizationType.ADAPTIVE, Qt.ItemDataRole.UserRole)
        )
        dialog.accept()

    QTimer.singleShot(0, edit_and_accept_dialog)
    button = widget.findChild(QPushButton)
    assert button is not None
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)

    assert comboboxes_enabled == [False]
    assert parameter.update_strategy == LocalizationType.GLOBAL
    assert pytest.approx(es_settings.enkf_truncation) == 0.3
    assert settings_changed == [True]


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
    assert widget._parameter_config[0].update_strategy == LocalizationType.GLOBAL
