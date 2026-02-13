from unittest.mock import Mock

import pytest
from pytestqt.qtbot import QtBot

import ert.gui
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import EnsembleSelector
from ert.gui.main import GUILogHandler
from ert.gui.simulation.evaluate_ensemble_panel import EvaluateEnsemblePanel
from ert.gui.simulation.experiment_panel import ExperimentPanel
from tests.ert.ui_tests.gui.conftest import (
    get_child,
)

from .conftest import (
    REALIZATION_FINISHED_SUCCESSFULLY,
    MockStorage,
)


@pytest.mark.usefixtures("copy_poly_case")
def test_that_the_run_experiments_button_is_disabled(qtbot):
    args = Mock()
    args.config = "poly.ert"
    gui, *_ = ert.gui.main._start_initial_gui_window(args, GUILogHandler())
    experiment_panel = get_child(gui, ExperimentPanel)
    evaluate_ensemble_panel = get_child(experiment_panel, EvaluateEnsemblePanel)
    assert not evaluate_ensemble_panel.isConfigurationValid()


def test_that_ensemble_select_contains_only_leaf_ensembles(
    qtbot: QtBot,
) -> None:
    notifier = ErtNotifier()
    notifier._storage = MockStorage()
    notifier._storage._setup_mocked_run(
        "mock_ensemble_parent",
        "mock_experiment",
        [
            REALIZATION_FINISHED_SUCCESSFULLY,
            REALIZATION_FINISHED_SUCCESSFULLY,
        ],
    )
    notifier._storage._setup_mocked_run(
        "mock_ensemble_child",
        "mock_experiment",
        [
            REALIZATION_FINISHED_SUCCESSFULLY,
            REALIZATION_FINISHED_SUCCESSFULLY,
        ],
    )
    notifier._storage._setup_mocked_run(
        "mock_ensemble_leaf",
        "mock_experiment",
        [
            REALIZATION_FINISHED_SUCCESSFULLY,
            REALIZATION_FINISHED_SUCCESSFULLY,
        ],
    )

    keys = list(notifier._storage._ensembles.keys())
    notifier._storage._ensembles[keys[1]]._index.prior_ensemble_id = keys[0]
    notifier._storage._ensembles[keys[2]]._index.prior_ensemble_id = keys[1]

    panel = EvaluateEnsemblePanel(
        run_path="",
        notifier=notifier,
    )

    qtbot.addWidget(panel)
    ensemble_selector = panel.findChild(EnsembleSelector)
    assert ensemble_selector.count() == 1
    assert ensemble_selector.itemText(0) == "mock_experiment : mock_ensemble_leaf"
