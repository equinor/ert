from os import path
import math
from unittest.mock import patch
from typing import List
from qtpy.QtWidgets import QApplication
from qtpy.QtCore import Qt
import pytest
from pydantic import BaseModel

from ert.ensemble_evaluator import identifiers as ids
from ert.ensemble_evaluator.snapshot import (
    PartialSnapshot,
    SnapshotBuilder,
)
from ert.ensemble_evaluator import state
from ert.ensemble_evaluator.event import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)
from ert_gui.simulation.run_dialog import RunDialog
from ert_gui.model.snapshot import FileRole


def test_success(runmodel, qtbot, mock_tracker):
    widget = RunDialog("poly.ert", runmodel)
    widget.show()
    qtbot.addWidget(widget)

    with patch("ert_gui.simulation.run_dialog.EvaluatorTracker") as tracker:
        tracker.return_value = mock_tracker([EndEvent(failed=False, failed_msg="")])
        widget.startSimulation()

    with qtbot.waitExposed(widget, timeout=30000):
        qtbot.waitUntil(lambda: widget._total_progress_bar.value() == 100)
        assert widget.done_button.isVisible()
        assert widget.done_button.text() == "Done"


def test_large_snapshot(runmodel, large_snapshot, qtbot, mock_tracker):
    widget = RunDialog("poly.ert", runmodel)
    widget.show()
    qtbot.addWidget(widget)

    with patch("ert_gui.simulation.run_dialog.EvaluatorTracker") as tracker:
        iter_0 = FullSnapshotEvent(
            snapshot=large_snapshot,
            phase_name="Foo",
            current_phase=0,
            total_phases=1,
            progress=0.5,
            iteration=0,
            indeterminate=False,
        )
        iter_1 = FullSnapshotEvent(
            snapshot=large_snapshot,
            phase_name="Foo",
            current_phase=0,
            total_phases=1,
            progress=0.5,
            iteration=1,
            indeterminate=False,
        )
        tracker.return_value = mock_tracker(
            [iter_0, iter_1, EndEvent(failed=False, failed_msg="")]
        )
        widget.startSimulation()

    with qtbot.waitExposed(widget, timeout=30000):
        qtbot.waitUntil(lambda: widget._total_progress_bar.value() == 100, timeout=5000)
        qtbot.mouseClick(widget.show_details_button, Qt.LeftButton)
        qtbot.waitUntil(lambda: widget._tab_widget.count() == 2, timeout=5000)


@pytest.mark.parametrize(
    "events,tab_widget_count",
    [
        pytest.param(
            [
                FullSnapshotEvent(
                    snapshot=(
                        SnapshotBuilder()
                        .add_step(step_id="0", status=state.STEP_STATE_UNKNOWN)
                        .add_job(
                            step_id="0",
                            job_id="0",
                            index="0",
                            name="job_0",
                            data={},
                            status=state.JOB_STATE_START,
                        )
                        .build(["0"], state.REALIZATION_STATE_UNKNOWN)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.25,
                    iteration=0,
                    indeterminate=False,
                ),
                SnapshotUpdateEvent(
                    partial_snapshot=PartialSnapshot(
                        SnapshotBuilder().build(
                            [], status=state.REALIZATION_STATE_FINISHED
                        )
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.5,
                    iteration=0,
                    indeterminate=False,
                ),
                EndEvent(failed=False, failed_msg=""),
            ],
            1,
            id="real_less_partial",
        ),
        pytest.param(
            [
                FullSnapshotEvent(
                    snapshot=(
                        SnapshotBuilder()
                        .add_step(step_id="0", status=state.STEP_STATE_UNKNOWN)
                        .add_job(
                            step_id="0",
                            job_id="0",
                            index="0",
                            name="job_0",
                            data={
                                ids.MAX_MEMORY_USAGE: 1000,
                                ids.CURRENT_MEMORY_USAGE: 500,
                            },
                            status=state.JOB_STATE_START,
                        )
                        .build(["0"], state.REALIZATION_STATE_UNKNOWN)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.25,
                    iteration=0,
                    indeterminate=False,
                ),
                SnapshotUpdateEvent(
                    partial_snapshot=PartialSnapshot(
                        SnapshotBuilder()
                        .add_step(step_id="0", status=state.STEP_STATE_SUCCESS)
                        .build(["0"], status=state.REALIZATION_STATE_FINISHED)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.5,
                    iteration=0,
                    indeterminate=False,
                ),
                EndEvent(failed=False, failed_msg=""),
            ],
            1,
            id="jobless_partial",
        ),
        pytest.param(
            [
                FullSnapshotEvent(
                    snapshot=(
                        SnapshotBuilder()
                        .add_step(step_id="0", status=state.STEP_STATE_UNKNOWN)
                        .add_job(
                            step_id="0",
                            job_id="0",
                            index="0",
                            name="job_0",
                            data={},
                            status=state.JOB_STATE_START,
                        )
                        .add_job(
                            step_id="0",
                            job_id="1",
                            index="1",
                            name="job_1",
                            data={},
                            status=state.JOB_STATE_START,
                        )
                        .build(["0", "1"], state.REALIZATION_STATE_UNKNOWN)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.25,
                    iteration=0,
                    indeterminate=False,
                ),
                SnapshotUpdateEvent(
                    partial_snapshot=PartialSnapshot(
                        SnapshotBuilder()
                        .add_step(step_id="0", status=state.STEP_STATE_SUCCESS)
                        .add_job(
                            step_id="0",
                            job_id="0",
                            index="0",
                            status=state.JOB_STATE_FINISHED,
                            name="job_0",
                            data={},
                        )
                        .build(["1"], status=state.REALIZATION_STATE_RUNNING)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.5,
                    iteration=0,
                    indeterminate=False,
                ),
                SnapshotUpdateEvent(
                    partial_snapshot=PartialSnapshot(
                        SnapshotBuilder()
                        .add_step(step_id="0", status=state.STEP_STATE_FAILURE)
                        .add_job(
                            step_id="0",
                            job_id="1",
                            index="1",
                            status=state.JOB_STATE_FAILURE,
                            name="job_1",
                            data={},
                        )
                        .build(["0"], status=state.REALIZATION_STATE_FAILED)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.5,
                    iteration=0,
                    indeterminate=False,
                ),
                EndEvent(failed=False, failed_msg=""),
            ],
            1,
            id="two_job_updates_over_two_partials",
        ),
        pytest.param(
            [
                FullSnapshotEvent(
                    snapshot=(
                        SnapshotBuilder()
                        .add_step(step_id="0", status=state.STEP_STATE_UNKNOWN)
                        .add_job(
                            step_id="0",
                            job_id="0",
                            index="0",
                            name="job_0",
                            data={},
                            status=state.JOB_STATE_START,
                        )
                        .build(["0"], state.REALIZATION_STATE_UNKNOWN)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.25,
                    iteration=0,
                    indeterminate=False,
                ),
                FullSnapshotEvent(
                    snapshot=(
                        SnapshotBuilder()
                        .add_step(step_id="0", status=state.STEP_STATE_UNKNOWN)
                        .add_job(
                            step_id="0",
                            job_id="0",
                            index="0",
                            name="job_0",
                            data={},
                            status=state.JOB_STATE_START,
                        )
                        .build(["0"], state.REALIZATION_STATE_UNKNOWN)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.5,
                    iteration=1,
                    indeterminate=False,
                ),
                EndEvent(failed=False, failed_msg=""),
            ],
            2,
            id="two_iterations",
        ),
    ],
)
def test_run_dialog(events, tab_widget_count, runmodel, qtbot, mock_tracker):
    widget = RunDialog("poly.ert", runmodel)
    widget.show()
    qtbot.addWidget(widget)

    with patch("ert_gui.simulation.run_dialog.EvaluatorTracker") as tracker:
        tracker.return_value = mock_tracker(events)
        widget.startSimulation()

    with qtbot.waitExposed(widget, timeout=30000):
        qtbot.mouseClick(widget.show_details_button, Qt.LeftButton)
        qtbot.waitUntil(
            lambda: widget._tab_widget.count() == tab_widget_count, timeout=5000
        )
        qtbot.waitUntil(lambda: widget.done_button.isVisible(), timeout=5000)
        qtbot.wait(3000)


def create_file_dialog_events(text_filepath: str) -> List[BaseModel]:
    return [
        FullSnapshotEvent(
            snapshot=(
                SnapshotBuilder()
                .add_step(step_id="0", status=state.STEP_STATE_UNKNOWN)
                .add_job(
                    step_id="0",
                    job_id="0",
                    index="0",
                    name="job_0",
                    stdout=path.join(
                        path.dirname(path.realpath(__file__)),
                        text_filepath,
                    ),
                    data={},
                    status=state.JOB_STATE_START,
                )
                .build(["0"], state.REALIZATION_STATE_UNKNOWN)
            ),
            phase_name="Foo",
            current_phase=0,
            total_phases=1,
            progress=0.25,
            iteration=0,
            indeterminate=False,
        ),
        SnapshotUpdateEvent(
            partial_snapshot=PartialSnapshot(
                SnapshotBuilder()
                .add_step(step_id="0", status=state.STEP_STATE_SUCCESS)
                .add_job(
                    step_id="0",
                    job_id="0",
                    index="0",
                    status=state.JOB_STATE_FINISHED,
                    name="job_0",
                    data={},
                )
                .build(["1"], status=state.REALIZATION_STATE_RUNNING)
            ),
            phase_name="Foo",
            current_phase=0,
            total_phases=1,
            progress=0.5,
            iteration=0,
            indeterminate=False,
        ),
        EndEvent(failed=False, failed_msg=""),
    ]


def create_file_dialog_params():
    testCaseTextFiles = [
        "stdout-short",
        "stdout-just-right",
        "stdout-long",
        "stdout-long-and-extra-wide",
    ]
    return list(
        map(
            lambda textFile: pytest.param(
                create_file_dialog_events(textFile),
                1,
                id=textFile,
            ),
            testCaseTextFiles,
        )
    )


@pytest.mark.parametrize(
    "events,tab_widget_count",
    create_file_dialog_params(),
)
def test_stdout_dialog(events, tab_widget_count, runmodel, qtbot, mock_tracker):
    runDialogWidget = RunDialog("poly.ert", runmodel)
    runDialogWidget.show()
    qtbot.addWidget(runDialogWidget)

    with patch("ert_gui.simulation.run_dialog.EvaluatorTracker") as tracker:
        tracker.return_value = mock_tracker(events)
        runDialogWidget.startSimulation()

    with qtbot.waitExposed(runDialogWidget, timeout=30000):
        qtbot.mouseClick(runDialogWidget.show_details_button, Qt.LeftButton)
        qtbot.waitUntil(
            lambda: runDialogWidget._tab_widget.count() == tab_widget_count,
            timeout=5000,
        )
        qtbot.waitUntil(runDialogWidget.done_button.isVisible, timeout=5000)
        qtbot.waitUntil(lambda: runDialogWidget._total_progress_bar.value() == 100)

        # need to click on the realization
        tabWidget = runDialogWidget._tab_widget
        realizationsWidget = tabWidget.currentWidget()
        currentJob = realizationsWidget._real_view.model().index(0, 0)
        assert currentJob.isValid()
        qtbot.mouseClick(
            realizationsWidget._real_view.viewport(),
            Qt.LeftButton,
            pos=realizationsWidget._real_view.visualRect(currentJob).center(),
        )

        # find STDOUT field in table and click it
        jobView = runDialogWidget._job_view
        snapshotModel = jobView.model()
        jobIndex = snapshotModel.index(0, 4)
        assert jobIndex.isValid()
        rect = jobView.visualRect(jobIndex)
        qtbot.mouseClick(
            runDialogWidget._job_view.viewport(), Qt.LeftButton, pos=rect.center()
        )

        # check that we have a file dialog widget
        selectedFile = jobIndex.data(FileRole)
        assert selectedFile is not None
        fileDialog = runDialogWidget._open_files[selectedFile]
        qtbot.addWidget(fileDialog)
        qtbot.waitUntil(fileDialog.isVisible)

        # check that text field is large enough to display contents, but not
        # "too large" (expressed as ratios of screen width and height)
        textField = fileDialog._view
        fontMetrics = textField.fontMetrics()
        lineHeight = math.ceil(fontMetrics.lineSpacing())
        charWidth = fontMetrics.averageCharWidth()
        sizeLongestLine = get_size_of_longest_line(selectedFile) * charWidth
        textHeight = get_number_of_lines(selectedFile) * lineHeight
        screenHeight = QApplication.primaryScreen().geometry().height()
        screenWidth = QApplication.primaryScreen().geometry().width()
        maxWidth = 1 / 3 * screenWidth
        maxHeight = 1 / 3 * screenHeight

        # sadly we have to wait for the dialog to be sized appropriately
        qtbot.wait(10)

        if textHeight < maxHeight:
            assert textField.height() >= textHeight
        else:
            assert textField.height() == pytest.approx(
                maxHeight, rel=0.05 * screenHeight
            )
        if sizeLongestLine < maxWidth:
            assert textField.width() >= sizeLongestLine
        else:
            assert textField.width() == pytest.approx(maxWidth, rel=0.05 * screenWidth)


def get_size_of_longest_line(filepath: str) -> int:
    with open(filepath, mode="r", encoding="utf-8") as filePointer:
        return max(map(len, filePointer.readlines()))


def get_number_of_lines(filepath: str) -> int:
    with open(filepath, mode="r", encoding="utf-8") as filePointer:
        return len(filePointer.readlines())
