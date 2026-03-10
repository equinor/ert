import fileinput
import shutil

import polars as pl
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QListWidget

from ert.gui.experiments import ExperimentPanel
from ert.gui.experiments.run_dialog import RunDialog
from ert.gui.experiments.view.update import UpdateLogTable, UpdateWidget
from ert.run_models import (
    EnsembleExperiment,
    EnsembleSmoother,
    MultipleDataAssimilation,
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

    status_list = update_widget._tab_widget.widget(0).findChild(QListWidget)

    count = sum(
        1
        for i in range(status_list.count())
        if "No active observations for update step" in status_list.item(i).text()
    )
    assert count == 1, "message is expected to appear just once on the list"


@pytest.fixture
def poly_case_with_autoscale_observations_config(source_root, tmp_path, run_experiment):
    _new_poly_example(source_root, tmp_path, 2)

    config_path = tmp_path / "poly.ert"

    with open(config_path, "a", encoding="utf-8") as f:
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
            assert row[status_index] == "nan"
            assert row[missing_realizations_index] == "0"
        else:
            assert row[status_index] == "Active"
            assert not row[missing_realizations_index]
