from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QPushButton

from ert.gui.ertwidgets import ClosableDialog, StringBox
from ert.gui.ertwidgets.ensembleselector import EnsembleSelector
from ert.gui.tools.load_results import LoadResultsPanel

from .conftest import (
    get_child,
    wait_for_child,
)


def test_validation(ensemble_experiment_has_run_no_failure, qtbot):
    gui = ensemble_experiment_has_run_no_failure
    ensemble_name = "iter-0"

    def handle_load_results_dialog():
        dialog = wait_for_child(gui, qtbot, ClosableDialog)
        panel = get_child(dialog, LoadResultsPanel)

        ensemble_selector = get_child(panel, EnsembleSelector)
        index = ensemble_selector.findText(ensemble_name, Qt.MatchFlag.MatchContains)
        ensemble_selector.setCurrentIndex(index)

        load_button = get_child(panel.parent(), QPushButton, name="Load")

        active_realizations = get_child(
            panel, StringBox, name="active_realizations_lrm"
        )
        default_value_active_reals = active_realizations.get_text
        active_realizations.setText(
            f"0-{ensemble_selector.selected_ensemble.ensemble_size + 1}"
        )

        assert not load_button.isEnabled()
        active_realizations.setText(default_value_active_reals)
        assert load_button.isEnabled()

        iterations_field = get_child(panel, StringBox, name="iterations_field_lrm")
        default_value_iteration = iterations_field.get_text
        iterations_field.setText("-10")

        assert not load_button.isEnabled()
        iterations_field.setText(default_value_iteration)
        assert load_button.isEnabled()

        dialog.close()

    QTimer.singleShot(1000, handle_load_results_dialog)
    load_results_tool = gui.tools["Load results manually"]
    load_results_tool.trigger()
