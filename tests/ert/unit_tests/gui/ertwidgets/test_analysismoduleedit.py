from collections.abc import Callable

import pytest
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication, QDialog, QPushButton
from pytestqt.qtbot import QtBot

from ert.config import ESSettings, LocalizationType
from ert.gui.ertwidgets.analysismoduleedit import AnalysisModuleEdit
from ert.gui.ertwidgets.analysismodulevariablespanel import AnalysisModuleVariablesPanel


def _open_dialog_and(
    qtbot: QtBot,
    widget: AnalysisModuleEdit,
    action: Callable[[QDialog, AnalysisModuleVariablesPanel], None],
) -> None:
    def interact() -> None:
        dialog = QApplication.activeModalWidget()
        assert isinstance(dialog, QDialog)
        panel = dialog.findChild(AnalysisModuleVariablesPanel)
        assert panel is not None
        action(dialog, panel)

    QTimer.singleShot(0, interact)
    button = widget.findChild(QPushButton)
    assert button is not None
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)


def _edit_settings(panel: AnalysisModuleVariablesPanel) -> None:
    panel._correlation_threshold = 0.7
    panel._enkf_truncation = 0.3
    panel._update_strategies["GEN_KW"] = LocalizationType.ADAPTIVE


def test_that_accepting_dialog_updates_es_settings_and_emits_update_strategies(
    qtbot: QtBot,
):
    es_settings = ESSettings()
    es_settings.localization_correlation_threshold = 0.5
    es_settings.enkf_truncation = 0.2
    widget = AnalysisModuleEdit(
        es_settings=es_settings,
        get_update_strategies=lambda: {"GEN_KW": LocalizationType.GLOBAL},
        ensemble_size=10,
    )
    qtbot.addWidget(widget)
    emitted: list[dict[str, LocalizationType]] = []
    widget.update_strategies_changed.connect(emitted.append)

    def edit_and_accept(dialog: QDialog, panel: AnalysisModuleVariablesPanel) -> None:
        _edit_settings(panel)
        dialog.accept()

    _open_dialog_and(qtbot, widget, edit_and_accept)

    assert pytest.approx(es_settings.localization_correlation_threshold) == 0.7
    assert pytest.approx(es_settings.enkf_truncation) == 0.3
    assert emitted == [{"GEN_KW": LocalizationType.ADAPTIVE}]


def test_that_rejecting_dialog_leaves_es_settings_unchanged_and_emits_nothing(
    qtbot: QtBot,
):
    es_settings = ESSettings()
    es_settings.localization_correlation_threshold = 0.5
    es_settings.enkf_truncation = 0.2
    widget = AnalysisModuleEdit(
        es_settings=es_settings,
        get_update_strategies=lambda: {"GEN_KW": LocalizationType.GLOBAL},
        ensemble_size=10,
    )
    qtbot.addWidget(widget)
    emitted: list[dict[str, LocalizationType]] = []
    widget.update_strategies_changed.connect(emitted.append)

    def edit_and_reject(dialog: QDialog, panel: AnalysisModuleVariablesPanel) -> None:
        _edit_settings(panel)
        dialog.reject()

    _open_dialog_and(qtbot, widget, edit_and_reject)

    assert pytest.approx(es_settings.localization_correlation_threshold) == 0.5
    assert pytest.approx(es_settings.enkf_truncation) == 0.2
    assert emitted == []


def test_that_dialog_is_initialized_from_latest_update_strategies(qtbot: QtBot):
    current_strategies = {"GEN_KW": LocalizationType.GLOBAL}
    widget = AnalysisModuleEdit(
        es_settings=ESSettings(),
        get_update_strategies=lambda: current_strategies,
        ensemble_size=10,
    )
    qtbot.addWidget(widget)
    current_strategies = {"GEN_KW": LocalizationType.ADAPTIVE}
    seen: list[dict[str, LocalizationType]] = []

    def record_and_reject(dialog: QDialog, panel: AnalysisModuleVariablesPanel) -> None:
        seen.append(dict(panel.update_strategies))
        dialog.reject()

    _open_dialog_and(qtbot, widget, record_and_reject)

    assert seen == [{"GEN_KW": LocalizationType.ADAPTIVE}]


def test_that_editing_dialog_does_not_mutate_provided_update_strategies(
    qtbot: QtBot,
):
    provided = {"GEN_KW": LocalizationType.GLOBAL}
    widget = AnalysisModuleEdit(
        es_settings=ESSettings(),
        get_update_strategies=lambda: provided,
        ensemble_size=10,
    )
    qtbot.addWidget(widget)

    def edit_and_accept(dialog: QDialog, panel: AnalysisModuleVariablesPanel) -> None:
        _edit_settings(panel)
        dialog.accept()

    _open_dialog_and(qtbot, widget, edit_and_accept)

    assert provided == {"GEN_KW": LocalizationType.GLOBAL}
