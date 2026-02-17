import fileinput

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox

from ert.gui.experiments import ExperimentPanel
from ert.run_models import (
    EnsembleExperiment,
    EnsembleSmoother,
    MultipleDataAssimilation,
)
from tests.ert.ui_tests.gui.conftest import get_child, open_gui_with_config


@pytest.mark.usefixtures("copy_poly_case")
def test_no_updateable_parameters(qtbot):
    with fileinput.input("poly.ert", inplace=True) as fin:
        for line in fin:
            if "GEN_KW COEFFS coeff_priors" in line:
                print(f"{line[:-1]} UPDATE:FALSE")
            else:
                print(line, end="")

    for gui in open_gui_with_config("poly.ert"):
        experiment_panel = get_child(gui, ExperimentPanel)
        simulation_mode_combo = get_child(experiment_panel, QComboBox)
        idx = simulation_mode_combo.findText(EnsembleSmoother.display_name())
        assert not (
            simulation_mode_combo.model().item(idx).flags() & Qt.ItemFlag.ItemIsEnabled
        )
        idx = simulation_mode_combo.findText(MultipleDataAssimilation.display_name())
        assert not (
            simulation_mode_combo.model().item(idx).flags() & Qt.ItemFlag.ItemIsEnabled
        )
        idx = simulation_mode_combo.findText(EnsembleExperiment.display_name())
        assert (
            simulation_mode_combo.model().item(idx).flags() & Qt.ItemFlag.ItemIsEnabled
        )
