import fileinput
import shutil
from pathlib import Path
from queue import SimpleQueue
from unittest.mock import Mock
from uuid import uuid4

import polars as pl
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QTextEdit

from _ert.events import (
    WorkflowBatchStartedEvent,
)
from ert.config import HookRuntime
from ert.gui.experiments import ExperimentPanel
from ert.gui.experiments.run_dialog import RunDialog
from ert.gui.experiments.view import IterationWidget
from ert.gui.experiments.view.update import UpdateLogTable, UpdateWidget
from ert.run_models import (
    EnsembleExperiment,
    EnsembleSmoother,
    MultipleDataAssimilation,
    RunModelAPI,
    RunModelUpdateBeginEvent,
)
from ert.run_models.manual_update import ManualUpdate
from tests.ert.ui_tests.gui.conftest import (
    _new_poly_example,
    get_child,
    open_gui_with_config,
)


@pytest.mark.usefixtures("copy_poly_case")
def test_no_updateable_parameters(qtbot):
    with fileinput.input("poly.ert", inplace=True) as fin:
        for line in fin:
            if "GEN_KW COEFFS coeff_priors" in line:
                print(f"{line[:-1]} UPDATE:FALSE")
            else:
                print(line, end="")

    with open_gui_with_config("poly.ert") as gui:
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


@pytest.fixture
def copy_run_to_current_test(tmp_path, monkeypatch):
    def _copy_case(src_path):
        monkeypatch.chdir(tmp_path)

        dst_path = tmp_path
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

    return _copy_case


def remove_responses_in_realization0(gui, indices_to_drop):
    """Removes responses in the first realization of the only ensemble.
    indices_to_drop are expected to correspond to observation indices that must
    be disabled."""
    ensemble_path = gui.notifier.storage._ensemble_path(
        next(iter(gui.notifier.storage._ensembles.keys()))
    )

    realization0_response_file_path = ensemble_path / "realization-0/gen_data.parquet"
    df = pl.read_parquet(realization0_response_file_path)

    assert len(df) == 10, "test setup is expected to have 10 responses"
    df = df.filter(~pl.Series(range(len(df))).is_in(indices_to_drop))
    df.write_parquet(realization0_response_file_path)

    gui.notifier.emitErtChange()


@pytest.fixture(scope="module")
def poly_example_with_missing_responses_dir(tmp_path_factory):
    return tmp_path_factory.mktemp(
        "poly_example_with_missing_responses_for_each_observation"
    )


@pytest.fixture(scope="module")
def setup_poly_case_with_missing_response_for_each_observation(
    source_root, run_experiment, poly_example_with_missing_responses_dir
):
    """
    Sets up a poly case with removed responses corresponding to all
    observations. Setup is used by several tests, so is created as a module
    fixture to save time.
    """
    path = poly_example_with_missing_responses_dir
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(path)
        _new_poly_example(source_root, path, 2)

        with open_gui_with_config(path / "poly.ert") as gui:
            run_experiment(EnsembleExperiment, gui)

            # removed indexes are all indexes poly example has observations for
            remove_responses_in_realization0(gui, indices_to_drop=[0, 2, 4, 6, 8])
            yield gui


@pytest.fixture
def poly_case_with_missing_response_for_each_observation(
    copy_run_to_current_test,
    setup_poly_case_with_missing_response_for_each_observation,
    poly_example_with_missing_responses_dir,
    tmp_path,
):
    copy_run_to_current_test(poly_example_with_missing_responses_dir)
    with open_gui_with_config(tmp_path / "poly.ert") as gui:
        yield gui


@pytest.mark.parametrize("update_method", ["ES Update", "EnIF Update (Experimental)"])
def test_that_report_table_is_displayed_on_no_active_observations(
    qtbot,
    run_experiment,
    update_method,
    poly_case_with_missing_response_for_each_observation,
):
    gui = poly_case_with_missing_response_for_each_observation

    run_experiment(
        ManualUpdate, gui, check_realizations=False, update_method=update_method
    )
    run_dialog = gui.findChildren(RunDialog)[-1]

    update_widget = run_dialog.findChild(UpdateWidget)
    assert update_widget._tab_widget.count() == 2
    assert update_widget._tab_widget.tabText(0) == "Status"
    assert update_widget._tab_widget.tabText(1) == "Report"

    status_text = update_widget._tab_widget.widget(0).findChild(QTextEdit)

    content = status_text.toPlainText()
    count = content.count("No active observations for update step")
    assert count == 1, "message is expected to appear just once"


@pytest.fixture
def poly_case_with_autoscale_observations_config(source_root, tmp_path, run_experiment):
    _new_poly_example(source_root, tmp_path, 2)

    config_path = tmp_path / "poly.ert"

    with Path(config_path).open("a", encoding="utf-8") as f:
        f.write("\nANALYSIS_SET_VAR OBSERVATIONS AUTO_SCALE *\n")

    with open_gui_with_config(config_path) as gui:
        run_experiment(EnsembleExperiment, gui)
        yield gui


def test_that_autoscale_tab_is_displayed_in_es_update(
    qtbot, poly_case_with_autoscale_observations_config, run_experiment
):
    gui = poly_case_with_autoscale_observations_config

    run_experiment(ManualUpdate, gui, check_realizations=False)

    run_dialog = gui.findChildren(RunDialog)[-1]
    update_widget = run_dialog.findChild(UpdateWidget)
    assert update_widget._tab_widget.count() == 3
    assert update_widget._tab_widget.tabText(0) == "Status"
    assert update_widget._tab_widget.tabText(1) == "Auto scale: *"
    assert update_widget._tab_widget.tabText(2) == "Report"

    update_log_table = update_widget.findChild(UpdateLogTable)
    expected_columns = [
        "Observation",
        "Index",
        "Cluster",
        "Nr components",
        "Scaling factor",
    ]

    for i, name in enumerate(expected_columns):
        assert update_log_table.horizontalHeaderItem(i).text() == name


@pytest.fixture
def poly_case_with_one_missing_response(source_root, tmp_path, run_experiment):
    """
    Sets up a poly case with missing response in realization 0 at indices
    corresponding to observations 2 and 4.
    """
    _new_poly_example(source_root, tmp_path, 2)
    config_path = tmp_path / "poly.ert"
    # ensures observations are not disabled due to unlucky responses
    with Path(config_path).open("a", encoding="utf-8") as f:
        f.write("\nRANDOM_SEED 1234\n")

    with open_gui_with_config(config_path) as gui:
        run_experiment(EnsembleExperiment, gui)
        # observations exist at response indices 0, 2, 4, 6, 8
        remove_responses_in_realization0(gui, indices_to_drop=[4, 8])
        yield gui


def test_that_report_table_is_displayed_on_missing_responses(
    qtbot, poly_case_with_one_missing_response, run_experiment
):
    gui = poly_case_with_one_missing_response

    run_experiment(ManualUpdate, gui, check_realizations=False)

    run_dialog = gui.findChildren(RunDialog)[-1]
    update_widget = run_dialog.findChild(UpdateWidget)
    assert update_widget._tab_widget.count() == 2
    assert update_widget._tab_widget.tabText(0) == "Status"
    assert update_widget._tab_widget.tabText(1) == "Report"

    update_log_table = update_widget._tab_widget.widget(1).findChild(UpdateLogTable)

    status_index = update_log_table.data.header.index("status")
    missing_realizations_index = update_log_table.data.header.index(
        "missing_realizations"
    )

    expected_disabled_observations = [2, 4]
    for index, row in enumerate(update_log_table.data.data):
        if index in expected_disabled_observations:
            assert row[status_index] == "nan", f"at row {index}"
            assert row[missing_realizations_index] == "0"
        else:
            assert row[status_index] == "Active", f"at row {index}"
            assert not row[missing_realizations_index]


def test_run_dialog_displays_workflow_tabs_between_iterations(qtbot, tmp_path):
    run_model_api = RunModelAPI(
        experiment_name="Ensemble smoother",
        supports_rerunning_failed_realizations=False,
        start_simulations_thread=lambda *_args, **_kwargs: None,
        cancel=lambda: None,
        has_failed_realizations=lambda: False,
    )
    notifier = Mock()

    run_dialog = RunDialog(
        "Running experiment",
        run_model_api,
        SimpleQueue(),
        notifier,
        output_path=tmp_path,
        run_path=tmp_path,
        storage_path=tmp_path,
    )
    qtbot.addWidget(run_dialog)

    run_dialog._on_event(
        WorkflowBatchStartedEvent(
            hook=HookRuntime.PRE_SIMULATION,
            iteration=0,
            workflow_names=["prepare_iteration_0"],
        )
    )
    run_dialog._on_event(
        WorkflowBatchStartedEvent(
            hook=HookRuntime.POST_SIMULATION,
            iteration=0,
            workflow_names=["collect_iteration_0"],
        )
    )
    run_dialog._on_event(
        WorkflowBatchStartedEvent(
            hook=HookRuntime.PRE_UPDATE,
            iteration=0,
            workflow_names=["pre_update_iteration_0"],
        )
    )
    run_dialog._on_event(RunModelUpdateBeginEvent(iteration=0, run_id=uuid4()))
    run_dialog._on_event(
        WorkflowBatchStartedEvent(
            hook=HookRuntime.POST_UPDATE,
            iteration=0,
            workflow_names=["post_update_iteration_0"],
        )
    )
    run_dialog._on_event(
        WorkflowBatchStartedEvent(
            hook=HookRuntime.PRE_SIMULATION,
            iteration=1,
            workflow_names=["prepare_iteration_1"],
        )
    )

    assert run_dialog._tab_widget.tabText(0) == "iteration-0"
    assert run_dialog._tab_widget.tabText(1) == "update-0"
    assert run_dialog._tab_widget.tabText(2) == "iteration-1"

    first_iteration_widget = run_dialog._tab_widget.widget(0)
    update_widget = run_dialog._tab_widget.widget(1)
    second_iteration_widget = run_dialog._tab_widget.widget(2)
    assert isinstance(first_iteration_widget, IterationWidget)
    assert isinstance(update_widget, IterationWidget)
    assert isinstance(second_iteration_widget, IterationWidget)

    assert first_iteration_widget._tab_widget.tabText(0) == "Pre-simulation workflows"
    assert first_iteration_widget._tab_widget.tabText(1) == "Post-simulation workflows"
    assert update_widget._tab_widget.tabText(0) == "Pre-update workflows"
    assert update_widget._tab_widget.tabText(1) == "Update"
    assert update_widget._tab_widget.tabText(2) == "Post-update workflows"
    assert second_iteration_widget._tab_widget.tabText(0) == "Pre-simulation workflows"

    first_pre_simulation_widget = run_dialog._select_or_create_workflow_tab(
        HookRuntime.PRE_SIMULATION, 0
    )
    second_pre_simulation_widget = run_dialog._select_or_create_workflow_tab(
        HookRuntime.PRE_SIMULATION, 1
    )
    assert first_pre_simulation_widget is not second_pre_simulation_widget
