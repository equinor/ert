import contextlib
import shutil

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QToolButton, QTreeView, QWidget

from ert.data import MeasuredData
from ert.gui.simulation.evaluate_ensemble_panel import EvaluateEnsemblePanel
from ert.gui.simulation.experiment_panel import ExperimentPanel
from ert.gui.simulation.manual_update_panel import ManualUpdatePanel
from ert.gui.simulation.run_dialog import RunDialog
from ert.gui.tools.manage_experiments import ManageExperimentsPanel
from ert.gui.tools.manage_experiments.storage_widget import StorageWidget
from ert.run_models.evaluate_ensemble import EvaluateEnsemble
from ert.run_models.manual_update import ManualUpdate
from ert.validation import rangestring_to_mask

from .conftest import get_child, wait_for_child


def test_manual_analysis_workflow(ensemble_experiment_has_run, qtbot):
    """This runs a full manual update workflow, first running ensemble experiment
    where some of the realizations fail, then doing an update before running an
    ensemble experiment again to calculate the forecast of the update.
    """
    gui = ensemble_experiment_has_run

    # Select correct experiment in the simulation panel
    experiment_panel = get_child(gui, ExperimentPanel)
    simulation_settings = get_child(experiment_panel, ManualUpdatePanel)
    simulation_mode_combo = get_child(experiment_panel, QComboBox)
    simulation_mode_combo.setCurrentText(ManualUpdate.name())

    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree("poly_out")

    # Click start simulation and agree to the message
    run_experiment = get_child(experiment_panel, QWidget, name="run_experiment")
    qtbot.mouseClick(run_experiment, Qt.MouseButton.LeftButton)
    # The Run dialog opens, wait until done appears, then click done
    run_dialog = wait_for_child(gui, qtbot, RunDialog)
    qtbot.waitUntil(lambda: run_dialog.is_simulation_done() == True, timeout=10000)
    qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)

    button_manage_experiments = gui.findChild(QToolButton, "button_Manage_experiments")
    assert button_manage_experiments
    qtbot.mouseClick(button_manage_experiments, Qt.MouseButton.LeftButton)
    experiments_panel = gui.findChild(ManageExperimentsPanel)
    assert experiments_panel

    # In the "create new case" tab, it should now contain "iter-1"
    experiments_panel.setCurrentIndex(0)
    current_tab = experiments_panel.currentWidget()
    assert current_tab
    assert current_tab.objectName() == "create_new_ensemble_tab"
    storage_widget = get_child(current_tab, StorageWidget)
    tree_view = get_child(storage_widget, QTreeView)
    tree_view.expandAll()

    model = tree_view.model()
    assert model is not None
    assert model.rowCount() == 1
    assert "iter-0_1" in model.index(1, 0, model.index(0, 0)).data(0)

    experiments_panel.close()

    simulation_settings = get_child(experiment_panel, EvaluateEnsemblePanel)
    simulation_mode_combo = get_child(experiment_panel, QComboBox)
    simulation_mode_combo.setCurrentText(EvaluateEnsemble.name())

    idx = simulation_settings._ensemble_selector.findData(
        "ensemble_experiment : iter-0_1",
        Qt.ItemDataRole.DisplayRole,
        Qt.MatchFlag.MatchStartsWith,
    )
    assert idx != -1
    simulation_settings._ensemble_selector.setCurrentIndex(idx)

    storage = gui.notifier.storage
    experiment = storage.get_experiment_by_name("ensemble_experiment")
    ensemble_prior = experiment.get_ensemble_by_name("iter-0")
    active_reals = list(ensemble_prior.get_realization_mask_with_responses())
    # Assert that some realizations failed
    assert not all(active_reals)
    assert active_reals == rangestring_to_mask(
        experiment_panel.get_experiment_arguments().realizations,
        20,
    )

    df_prior = ensemble_prior.load_all_gen_kw_data()
    ensemble_posterior = experiment.get_ensemble_by_name("iter-0_1")
    df_posterior = ensemble_posterior.load_all_gen_kw_data()

    # Making sure measured data works with failed realizations
    MeasuredData(experiment.get_ensemble_by_name("iter-0"), ["POLY_OBS"])

    # We expect that ERT's update step lowers the
    # generalized variance for the parameters.
    assert (
        0
        < np.linalg.det(df_posterior.cov().to_numpy())
        < np.linalg.det(df_prior.cov().to_numpy())
    )
