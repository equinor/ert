from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QPushButton

from ert.gui.ertwidgets import ClosableDialog, EnsembleSelector, StringBox, TextBox
from ert.gui.tools.load_results import LoadResultsPanel

from .conftest import get_child, wait_for_child


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

        run_path_edit = get_child(panel, TextBox, name="run_path_edit_lrm")
        assert run_path_edit.isEnabled()
        valid_text = run_path_edit.get_text
        assert "<IENS>" in valid_text

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

        run_path_edit.setText(valid_text.replace("<IENS>", "<IES>"))
        assert not load_button.isEnabled()
        run_path_edit.setText(valid_text)
        assert load_button.isEnabled()

        dialog.close()

    QTimer.singleShot(1000, handle_load_results_dialog)
    gui.load_results_tool.trigger()
