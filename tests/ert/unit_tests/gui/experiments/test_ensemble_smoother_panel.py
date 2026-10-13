from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QPushButton,
)
from pytestqt.qtbot import QtBot

from ert.config import GenKwConfig
from ert.config.analysis_config import AnalysisConfig
from ert.config.distribution import RawSettings
from ert.config.parameter_config import LocalizationType
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.experiments.ensemble_smoother_panel import EnsembleSmootherPanel

from .conftest import MockStorage


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

    assert (
        panel.get_experiment_arguments().parameter_configuration[0].update_strategy
        == LocalizationType.ADAPTIVE
    )
