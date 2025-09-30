from unittest.mock import Mock

import pytest
from pytestqt.qtbot import QtBot

import ert.gui
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets.ensembleselector import EnsembleSelector
from ert.gui.ertwidgets.stringbox import StringBox
from ert.gui.main import GUILogHandler
from ert.gui.simulation.evaluate_ensemble_panel import EvaluateEnsemblePanel
from ert.gui.simulation.experiment_panel import ExperimentPanel
from tests.ert.ui_tests.gui.conftest import (
    get_child,
)

from .conftest import (
    REALIZATION_FAILED_DURING_EVALUATION,
    REALIZATION_FINISHED_SUCCESSFULLY,
    REALIZATION_ONLY_PARAMETERS,
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


@pytest.mark.usefixtures("copy_poly_case")
def test_evaluate_ensemble_active_realizations_resets_to_all_realizations_with_parameters_when_all_realizations_are_successful_JONAK(  # noqa
    qtbot: QtBot,
):
    """
    The evaluate_ensemble panel automatically selects the realizations with
    failed state from the selected ensemble as active realizations. This caused
    issues if all realizations in the selected ensemble had successful state,
    and it defaulted to empty string, causing a validation error stating the
    active realizations cannot be empty. This test makes sure that if this situation
    occurs, we instead select all realizations with parameters.
    """
    notifier = ErtNotifier()
    notifier._storage = MockStorage()
    ensemble_size = 100
    ensemble_state = [REALIZATION_ONLY_PARAMETERS] * ensemble_size
    notifier._storage._setup_mocked_run(
        "mock_ensemble", "mock_experiment", ensemble_state
    )
    ensemble = notifier.current_ensemble
    assert ensemble is not None
    evaluate_ensemble_panel = EvaluateEnsemblePanel(ensemble_size, "", notifier)
    qtbot.add_widget(evaluate_ensemble_panel)
    realization_selector = evaluate_ensemble_panel.findChild(StringBox)
    ensemble_selector = evaluate_ensemble_panel.findChild(EnsembleSelector)

    # The selector defaults to all realizations, and is valid
    assert realization_selector.isValid()
    assert realization_selector.text() == f"0-{ensemble_size - 1}"

    # update ensemble to have half of ensemble finish successfully
    ensemble._storage_state = [REALIZATION_FINISHED_SUCCESSFULLY] * (
        ensemble_size // 2
    ) + [REALIZATION_FAILED_DURING_EVALUATION] * (ensemble_size // 2)
    notifier.ertChanged.emit()
    qtbot.wait_signal(ensemble_selector.ensemble_populated)
    assert realization_selector.isValid()
    assert realization_selector.text() == "50-99"

    # update realizations to have remaining realizations finish successfully
    ensemble._storage_state = [REALIZATION_FINISHED_SUCCESSFULLY] * ensemble_size
    notifier.ertChanged.emit()
    qtbot.wait_signal(ensemble_selector.ensemble_populated)
    assert realization_selector.isValid()
    assert realization_selector.text() == f"0-{ensemble_size - 1}"
