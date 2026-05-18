import fileinput
import shutil
from io import BytesIO
from pathlib import Path
from textwrap import dedent

import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import pytest
from PyQt6.QtCore import QBuffer, QByteArray, QIODeviceBase, Qt
from PyQt6.QtWidgets import QComboBox, QTextEdit, QWidget
from pytestqt.qtbot import QtBot

from _ert.events import (
    WorkflowStatus,
)
from ert.config import HookRuntime
from ert.gui.experiments import ExperimentPanel
from ert.gui.experiments.run_dialog import RunDialog
from ert.gui.experiments.view import RealizationWidget, TabGroupWidget
from ert.gui.experiments.view.update import UpdateLogTable, UpdateWidget
from ert.gui.experiments.view.workflow import (
    HEADER_TO_COLUMN,
    WorkflowWidget,
    workflow_tab_title,
)
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

ALL_WORKFLOW_HOOKS = (
    (HookRuntime.PRE_EXPERIMENT, "pre_experiment"),
    (HookRuntime.PRE_SIMULATION, "pre_simulation"),
    (HookRuntime.POST_SIMULATION, "post_simulation"),
    (HookRuntime.PRE_FIRST_UPDATE, "pre_first_update"),
    (HookRuntime.PRE_UPDATE, "pre_update"),
    (HookRuntime.POST_UPDATE, "post_update"),
    (HookRuntime.POST_EXPERIMENT, "post_experiment"),
)


def _tab_texts(widget) -> list[str]:
    return [widget.tabText(index) for index in range(widget.count())]


def _widget_for_tab_text(widget, tab_text: str):
    for index in range(widget.count()):
        if widget.tabText(index) == tab_text:
            return widget.widget(index)
    raise AssertionError(f"Did not find tab with text: {tab_text}")


def _append_all_workflow_hooks(config_path: Path) -> None:
    config_dir = config_path.parent
    (config_dir / "print_job").write_text("EXECUTABLE echo\n", encoding="utf-8")

    config_lines = ["", "LOAD_WORKFLOW_JOB print_job PRINT"]
    for hook, workflow_name in ALL_WORKFLOW_HOOKS:
        (config_dir / workflow_name).write_text(
            dedent(
                f"""\
                PRINT {workflow_name}
                """
            ),
            encoding="utf-8",
        )
        config_lines.extend(
            [
                f"LOAD_WORKFLOW {workflow_name} {workflow_name}",
                f"HOOK_WORKFLOW {workflow_name} {hook.value}",
            ]
        )

    with config_path.open("a", encoding="utf-8") as config_file:
        config_file.write("\n".join(config_lines) + "\n")


def _assert_finished_workflow_widget(
    widget: WorkflowWidget,
    workflow_name: str,
) -> None:
    assert widget._status_label.text() == f"{workflow_tab_title(widget.hook)} finished"
    assert widget._table.rowCount() == 1
    name_item = widget._table.item(0, HEADER_TO_COLUMN["WORKFLOW"])
    assert name_item is not None
    assert name_item.text() == workflow_name
    status_item = widget._table.item(0, HEADER_TO_COLUMN["STATUS"])
    assert status_item is not None
    assert status_item.text() == WorkflowStatus.FINISHED.value


def _render_widget_to_figure(widget: RunDialog):
    pixmap = widget.grab()
    byte_array = QByteArray()
    buffer = QBuffer(byte_array)
    assert buffer.open(QIODeviceBase.OpenModeFlag.WriteOnly)
    assert pixmap.save(buffer, "PNG")
    image = plt.imread(BytesIO(byte_array.data()), format="png")

    figure, axis = plt.subplots(
        figsize=(image.shape[1] / 100, image.shape[0] / 100),
        dpi=100,
    )
    axis.imshow(image)
    axis.axis("off")
    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return figure


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


def remove_responses_in_realization0(gui: QWidget, indices_to_drop: list[int]):
    """Removes responses in the first realization of the only ensemble.
    indices_to_drop are expected to correspond to observation indices that must
    be disabled.
    """
    ensemble_path = gui.notifier.storage._ensemble_path(
        next(iter(gui.notifier.storage._ensembles.keys()))
    )

    realization0_response_file_path = ensemble_path / "realization-0/gen_data.parquet"
    df = pl.read_parquet(realization0_response_file_path)

    assert len(df) == 10, "test setup is expected to have 10 responses"
    df = df.filter(~pl.Series(range(len(df))).is_in(indices_to_drop))
    df.write_parquet(realization0_response_file_path)

    gui.notifier.emitErtChange()


def change_responses_in_realization0(gui, indices_to_change):
    """Changes report_step to new value for rows in indices_to_change."""
    ensemble_path = gui.notifier.storage._ensemble_path(
        next(iter(gui.notifier.storage._ensembles.keys()))
    )

    realization0_response_file_path = ensemble_path / "realization-0/gen_data.parquet"
    df = pl.read_parquet(realization0_response_file_path)
    assert len(df) == 10, "test setup is expected to have 10 responses"

    df = df.with_columns(
        pl.when(pl.Series(range(len(df))).is_in(indices_to_change))
        .then(666)
        .otherwise(pl.col("report_step"))
        .alias("report_step")
    )
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
def poly_case_with_autoscale_observations_config(
    source_root, tmp_path, monkeypatch, run_experiment
):
    monkeypatch.chdir(tmp_path)
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
def poly_case_with_one_missing_response(
    source_root, tmp_path, monkeypatch, run_experiment
):
    """
    Sets up a poly case with missing response in realization 0 at indices
    corresponding to observations 2 and 4.
    """
    monkeypatch.chdir(tmp_path)
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

    response_mean_index = update_log_table.data.header.index("response_mean")
    status_index = update_log_table.data.header.index("status")
    missing_realizations_index = update_log_table.data.header.index(
        "missing_realizations"
    )

    expected_disabled_observations = [2, 4]
    for obs_index, row in enumerate(update_log_table.data.data):
        if obs_index in expected_disabled_observations:
            assert row[status_index] == "nan", f"at row {obs_index}"
            assert row[response_mean_index] is not None, f"at row {obs_index}"
            response_index = obs_index * 2
            msg = (
                "0: no response matched observation data: "
                f"response_key=POLY_RES, report_step=0, index={response_index}"
            )
            assert row[missing_realizations_index] == msg
        else:
            assert row[status_index] == "Active", f"at row {obs_index}"
            assert row[response_mean_index] is not None, f"at row {obs_index}"
            assert not row[missing_realizations_index]


@pytest.fixture(
    params=[range(10), range(8)],
    ids=[
        "all responses mismatched observation in realization 0",
        "one response has a match in observation in realization 0",
    ],
)
def poly_case_with_changed_response_on_match_key(
    source_root, tmp_path, monkeypatch, run_experiment, request
):
    """Sets up a poly case with all responses changed on match key in realization 0."""
    monkeypatch.chdir(tmp_path)
    _new_poly_example(source_root, tmp_path, 2)
    config_path = tmp_path / "poly.ert"
    # ensures observations are not disabled due to unlucky responses
    with Path(config_path).open("a", encoding="utf-8") as f:
        f.write("\nRANDOM_SEED 1234\n")

    with open_gui_with_config(config_path) as gui:
        run_experiment(EnsembleExperiment, gui)
        change_responses_in_realization0(gui, indices_to_change=request.param)
        yield gui


def test_that_response_mean_is_calculated_on_mismatched_responses(
    qtbot, poly_case_with_changed_response_on_match_key, run_experiment, request
):
    gui = poly_case_with_changed_response_on_match_key
    run_experiment(ManualUpdate, gui, check_realizations=False)

    run_dialog = gui.findChildren(RunDialog)[-1]
    update_widget = run_dialog.findChild(UpdateWidget)
    update_log_table = update_widget._tab_widget.widget(1).findChild(UpdateLogTable)
    response_mean_index = update_log_table.data.header.index("response_mean")

    for obs_index, row in enumerate(update_log_table.data.data):
        assert not np.isnan(row[response_mean_index]), f"at row {obs_index}"


@pytest.mark.mpl_image_compare(tolerance=10.0)
@pytest.mark.skip_mac_ci  # test is slow
@pytest.mark.filterwarnings(
    "ignore:Use of legacy_ertscript_workflow is deprecated.*:DeprecationWarning"
)
def test_that_esmda_with_all_workflows_produces_expected_tabs(
    qtbot: QtBot, tmp_path, source_root, run_experiment
):
    _new_poly_example(source_root, tmp_path, 3)
    config_path = tmp_path / "poly.ert"
    _append_all_workflow_hooks(config_path)

    with open_gui_with_config(config_path) as gui:
        qtbot.addWidget(gui)
        run_experiment(
            MultipleDataAssimilation,
            gui,
            check_realizations=False,
        )

        run_dialog = gui.findChildren(RunDialog)[-1]
        assert run_dialog.is_experiment_done() is True
        assert (
            run_dialog._total_progress_label.text()
            == "Total progress 100% — Experiment completed."
        )

        assert _tab_texts(run_dialog._tab_widget) == [
            "Pre-experiment workflows",
            "iteration-0",
            "update-0",
            "iteration-1",
            "update-1",
            "iteration-2",
            "update-2",
            "iteration-3",
            "Post-experiment workflows",
        ]
        assert (
            run_dialog._tab_widget.tabText(run_dialog._tab_widget.currentIndex())
            == "Post-experiment workflows"
        )

        pre_experiment_widget = _widget_for_tab_text(
            run_dialog._tab_widget, "Pre-experiment workflows"
        )
        post_experiment_widget = _widget_for_tab_text(
            run_dialog._tab_widget, "Post-experiment workflows"
        )
        assert isinstance(pre_experiment_widget, WorkflowWidget)
        assert isinstance(post_experiment_widget, WorkflowWidget)
        _assert_finished_workflow_widget(pre_experiment_widget, "pre_experiment")
        _assert_finished_workflow_widget(post_experiment_widget, "post_experiment")

        for iteration in range(4):
            iteration_widget = _widget_for_tab_text(
                run_dialog._tab_widget, f"iteration-{iteration}"
            )
            assert isinstance(iteration_widget, TabGroupWidget)
            assert set(_tab_texts(iteration_widget._tab_widget)) == {
                "Run",
                "Pre-simulation workflows",
                "Post-simulation workflows",
            }

            realization_widget = _widget_for_tab_text(
                iteration_widget._tab_widget, "Run"
            )
            assert isinstance(realization_widget, RealizationWidget)
            assert realization_widget._real_view.model().rowCount() == 3

            pre_simulation_widget = _widget_for_tab_text(
                iteration_widget._tab_widget, "Pre-simulation workflows"
            )
            post_simulation_widget = _widget_for_tab_text(
                iteration_widget._tab_widget, "Post-simulation workflows"
            )
            assert isinstance(pre_simulation_widget, WorkflowWidget)
            assert isinstance(post_simulation_widget, WorkflowWidget)
            _assert_finished_workflow_widget(pre_simulation_widget, "pre_simulation")
            _assert_finished_workflow_widget(post_simulation_widget, "post_simulation")

        for iteration in range(3):
            update_iteration_widget = _widget_for_tab_text(
                run_dialog._tab_widget, f"update-{iteration}"
            )
            assert isinstance(update_iteration_widget, TabGroupWidget)

            expected_tabs = {
                "Update",
                "Pre-update workflows",
                "Post-update workflows",
            }
            if iteration == 0:
                expected_tabs.add("Pre-first-update workflows")

            assert set(_tab_texts(update_iteration_widget._tab_widget)) == expected_tabs

            update_widget = _widget_for_tab_text(
                update_iteration_widget._tab_widget, "Update"
            )
            assert isinstance(update_widget, UpdateWidget)

            pre_update_widget = _widget_for_tab_text(
                update_iteration_widget._tab_widget, "Pre-update workflows"
            )
            post_update_widget = _widget_for_tab_text(
                update_iteration_widget._tab_widget, "Post-update workflows"
            )
            assert isinstance(pre_update_widget, WorkflowWidget)
            assert isinstance(post_update_widget, WorkflowWidget)
            _assert_finished_workflow_widget(pre_update_widget, "pre_update")
            _assert_finished_workflow_widget(post_update_widget, "post_update")

            if iteration == 0:
                pre_first_update_widget = _widget_for_tab_text(
                    update_iteration_widget._tab_widget,
                    "Pre-first-update workflows",
                )
                assert isinstance(pre_first_update_widget, WorkflowWidget)
                _assert_finished_workflow_widget(
                    pre_first_update_widget,
                    "pre_first_update",
                )

        return _render_widget_to_figure(run_dialog)
