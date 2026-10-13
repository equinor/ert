from unittest.mock import Mock

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPushButton,
)
from pytestqt.qtbot import QtBot

from ert.config import GenKwConfig
from ert.config.analysis_config import AnalysisConfig
from ert.config.distribution import RawSettings
from ert.config.parameter_config import LocalizationType, ParameterConfig
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.experiments.ensemble_smoother_panel import EnsembleSmootherPanel
from ert.run_models.ensemble_smoother import DEPRECATION_MESSAGE

from .conftest import MockStorage


def test_that_ensemble_smoother_panel_shows_deprecation_warning(qtbot: QtBot):
    notifier = ErtNotifier()
    notifier._storage = MockStorage()
    panel = EnsembleSmootherPanel(
        analysis_config=AnalysisConfig(minimum_required_realizations=1),
        parameter_configuration=[
            Mock(
                spec=ParameterConfig,
                update_strategy=LocalizationType.GLOBAL,
                type="gen_kw",
            )
        ],
        runpath="",
        notifier=notifier,
        active_realizations=[True, True],
        config_num_realization=2,
    )
    qtbot.addWidget(panel)

    warning = panel.findChild(QLabel, "ensemble_smoother_deprecation_warning")

    assert DEPRECATION_MESSAGE in warning.text()
    assert "Single update" in warning.text()


def test_that_update_strategy_edited_in_dialog_is_used_in_experiment_arguments(
    qtbot: QtBot,
):
    parameter = GenKwConfig(
        name="configured",
        distribution=RawSettings(),
        update_strategy=LocalizationType.GLOBAL,
    )
    notifier = ErtNotifier()
    notifier._storage = MockStorage()
    panel = EnsembleSmootherPanel(
        analysis_config=AnalysisConfig(minimum_required_realizations=1),
        parameter_configuration=[parameter],
        runpath="",
        notifier=notifier,
        active_realizations=[True] * 3,
        config_num_realization=3,
    )
    qtbot.addWidget(panel)

    def select_adaptive_strategy_and_save() -> None:
        dialog = QApplication.activeModalWidget()
        assert isinstance(dialog, QDialog)
        gen_kw_selector = dialog.findChildren(QComboBox)[0]
        gen_kw_selector.setCurrentIndex(
            gen_kw_selector.findData(
                LocalizationType.ADAPTIVE, Qt.ItemDataRole.UserRole
            )
        )
        buttons = dialog.findChild(QDialogButtonBox)
        assert buttons is not None
        save_button = buttons.button(QDialogButtonBox.StandardButton.Save)
        assert save_button is not None
        qtbot.mouseClick(save_button, Qt.MouseButton.LeftButton)

    QTimer.singleShot(0, select_adaptive_strategy_and_save)
    edit_button = panel.findChild(QPushButton, "analysis_variables_popup_button")
    assert edit_button is not None
    qtbot.mouseClick(edit_button, Qt.MouseButton.LeftButton)

    assert panel.get_experiment_arguments().parameter_configuration == [
        parameter.model_copy(update={"update_strategy": LocalizationType.ADAPTIVE})
    ]
