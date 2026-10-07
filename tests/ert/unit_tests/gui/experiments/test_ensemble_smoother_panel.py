from unittest.mock import Mock

from PyQt6.QtWidgets import QLabel
from pytestqt.qtbot import QtBot

from ert.config.analysis_config import AnalysisConfig
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
