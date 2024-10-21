import fileinput

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox

from ert.gui.simulation.experiment_panel import ExperimentPanel
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
        idx = simulation_mode_combo.findText("Ensemble smoother")
        assert not (
            simulation_mode_combo.model().item(idx).flags() & Qt.ItemFlag.ItemIsEnabled
        )
        idx = simulation_mode_combo.findText("Multiple data assimilation")
        assert not (
            simulation_mode_combo.model().item(idx).flags() & Qt.ItemFlag.ItemIsEnabled
        )
        idx = simulation_mode_combo.findText("Iterated ensemble smoother")
        assert not (
            simulation_mode_combo.model().item(idx).flags() & Qt.ItemFlag.ItemIsEnabled
        )
        idx = simulation_mode_combo.findText("Ensemble experiment")
        assert (
            simulation_mode_combo.model().item(idx).flags() & Qt.ItemFlag.ItemIsEnabled
        )
