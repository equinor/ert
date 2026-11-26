from PyQt6.QtWidgets import QComboBox
from pytestqt.qtbot import QtBot

from ert.config import GenKwConfig
from ert.config.analysis_config import AnalysisConfig
from ert.config.distribution import RawSettings
from ert.config.parameter_config import LocalizationType
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.experiments._update_strategy_summary_widget import (
    UpdateStrategySummaryWidget,
)
from ert.gui.experiments.ensemble_smoother_panel import EnsembleSmootherPanel
from ert.mode_definitions import ES_MDA_MODE

from .conftest import MockStorage


def _create_panel(qtbot: QtBot, num_realizations: int = 3) -> EnsembleSmootherPanel:
    notifier = ErtNotifier()
    notifier._storage = MockStorage()
    panel = EnsembleSmootherPanel(
        analysis_config=AnalysisConfig(minimum_required_realizations=1),
        parameter_configuration=[
            GenKwConfig(
                name="configured",
                distribution=RawSettings(),
                update_strategy=LocalizationType.GLOBAL,
            )
        ],
        runpath="",
        notifier=notifier,
        active_realizations=[True] * num_realizations,
        config_num_realization=num_realizations,
    )
    qtbot.addWidget(panel)
    return panel


def test_that_ensemble_smoother_panel_runs_es_mda_with_a_single_unit_weight(
    qtbot: QtBot,
):
    arguments = _create_panel(qtbot).get_experiment_arguments()

    assert arguments.mode == ES_MDA_MODE
    assert arguments.weights == "1"
    assert arguments.prior_ensemble_id is None


def test_that_update_strategy_chosen_in_summary_is_used_in_experiment_arguments(
    qtbot: QtBot,
):
    panel = _create_panel(qtbot)
    summary = panel.findChild(UpdateStrategySummaryWidget)
    assert summary is not None
    selector = summary.cellWidget(0, 0)
    assert isinstance(selector, QComboBox)

    selector.setCurrentIndex(selector.findData(LocalizationType.ADAPTIVE))

    assert (
        panel.get_experiment_arguments().parameter_configuration[0].update_strategy
        == LocalizationType.ADAPTIVE
    )
