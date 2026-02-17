from unittest.mock import Mock

import pytest

import ert.gui
from ert.gui.experiments import ExperimentPanel
from ert.gui.experiments.evaluate_ensemble_panel import EvaluateEnsemblePanel
from ert.gui.main import GUILogHandler
from tests.ert.ui_tests.gui.conftest import (
    get_child,
)


@pytest.mark.usefixtures("copy_poly_case")
def test_that_the_run_experiments_button_is_disabled(qtbot):
    args = Mock()
    args.config = "poly.ert"
    gui, *_ = ert.gui.main._start_initial_gui_window(args, GUILogHandler())
    experiment_panel = get_child(gui, ExperimentPanel)
    evaluate_ensemble_panel = get_child(experiment_panel, EvaluateEnsemblePanel)
    assert not evaluate_ensemble_panel.isConfigurationValid()
