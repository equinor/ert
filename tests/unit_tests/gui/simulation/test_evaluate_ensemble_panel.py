from unittest.mock import Mock

import pytest

import ert.gui
from ert.gui.main import GUILogHandler
from ert.gui.simulation.evaluate_ensemble_panel import EvaluateEnsemblePanel
from ert.gui.simulation.simulation_panel import SimulationPanel
from tests.unit_tests.gui.conftest import get_child


@pytest.mark.usefixtures("copy_poly_case")
def test_that_the_run_experiments_button_is_disabled(qtbot):
    args = Mock()
    args.config = "poly.ert"
    gui, *_ = ert.gui.main._start_initial_gui_window(args, GUILogHandler())
    simulation_panel = get_child(gui, SimulationPanel)
    evaluate_ensemble_panel = get_child(simulation_panel, EvaluateEnsemblePanel)
    assert not evaluate_ensemble_panel.isConfigurationValid()
