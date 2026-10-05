import pytest

from ert.gui.experiments.view import WorkflowLogWidget
from ert.gui.experiments.view.workflow_log import (
    NO_ITERATION_LABEL,
    NO_OUTPUT_PLACEHOLDER,
)
from ert.workflow_runner import WorkflowJobStatus
from tests.ert.unit_tests.gui.experiments.conftest import make_workflow_event


@pytest.fixture
def widget(qtbot):
    widget = WorkflowLogWidget()
    qtbot.addWidget(widget)
    return widget


def column_values(widget: WorkflowLogWidget, column: int) -> list[str]:
    return [
        widget._table.item(row, column).text()
        for row in range(widget._table.rowCount())
    ]


def pick_iteration(widget: WorkflowLogWidget, index: int) -> None:
    """Select an iteration the way a user would, via the combo box."""
    widget._iteration_selector.setCurrentIndex(index)
    widget._iteration_selector.activated.emit(index)


def test_that_each_workflow_event_adds_one_table_row(widget):
    widget.add_event(make_workflow_event(job_name="first"))
    widget.add_event(make_workflow_event(job_name="second", job_index=1))

    assert widget._table.rowCount() == 2
    assert column_values(widget, 2) == ["first", "second"]


def test_that_job_arguments_are_shown_next_to_job_name(widget):
    widget.add_event(make_workflow_event(job_name="echo", arguments=["a", "b"]))

    assert column_values(widget, 2) == ["echo(a, b)"]


def test_that_iteration_selector_lists_every_iteration_seen_in_events(widget):
    widget.add_event(make_workflow_event(iteration=1))
    widget.add_event(make_workflow_event(iteration=0))
    widget.add_event(make_workflow_event(iteration=1))

    labels = [
        widget._iteration_selector.itemText(i)
        for i in range(widget._iteration_selector.count())
    ]
    assert labels == ["Iteration 0", "Iteration 1"]


def test_that_selecting_iteration_filters_table_to_its_jobs(widget):
    widget.add_event(make_workflow_event(iteration=0, job_name="job_in_iter_0"))
    widget.add_event(make_workflow_event(iteration=1, job_name="job_in_iter_1"))

    pick_iteration(widget, 0)
    assert column_values(widget, 2) == ["job_in_iter_0"]

    pick_iteration(widget, 1)

    assert column_values(widget, 2) == ["job_in_iter_1"]


def test_that_events_for_selected_iteration_are_appended_live(widget):
    widget.add_event(make_workflow_event(iteration=0, job_name="first"))
    widget.add_event(make_workflow_event(iteration=1, job_name="other"))
    pick_iteration(widget, 1)

    widget.add_event(make_workflow_event(iteration=1, job_name="second"))

    assert column_values(widget, 2) == ["other", "second"]


def test_that_selector_follows_newest_iteration_until_user_picks_one(widget):
    widget.add_event(make_workflow_event(iteration=0, job_name="job_in_iter_0"))
    widget.add_event(make_workflow_event(iteration=1, job_name="job_in_iter_1"))

    assert widget._iteration_selector.currentText() == "Iteration 1"
    assert column_values(widget, 2) == ["job_in_iter_1"]

    pick_iteration(widget, 0)
    widget.add_event(make_workflow_event(iteration=2, job_name="job_in_iter_2"))

    assert widget._iteration_selector.currentText() == "Iteration 0"
    assert column_values(widget, 2) == ["job_in_iter_0"]


def test_that_events_without_iteration_are_listed_first_in_selector(widget):
    widget.add_event(make_workflow_event(iteration=1, job_name="job_in_iter_1"))
    widget.add_event(
        make_workflow_event(iteration=None, hook="PRE_EXPERIMENT", job_name="early_job")
    )

    assert widget._iteration_selector.itemText(0) == NO_ITERATION_LABEL
    assert widget._iteration_selector.itemData(0) is None


def test_that_selecting_pre_post_experiment_entry_shows_jobs_without_iteration(
    widget,
):
    widget.add_event(make_workflow_event(iteration=1, job_name="job_in_iter_1"))
    widget.add_event(
        make_workflow_event(iteration=None, hook="PRE_EXPERIMENT", job_name="early_job")
    )

    widget._iteration_selector.setCurrentIndex(0)

    assert column_values(widget, 0) == ["PRE_EXPERIMENT"]
    assert column_values(widget, 2) == ["early_job"]


def test_that_post_experiment_events_are_grouped_under_pre_post_experiment(widget):
    """POST_EXPERIMENT hooks run once for the whole experiment, but by then an
    ensemble already exists, so the event carries a real iteration number. It
    should still land under "Pre/post experiment", not under that iteration's
    tab alongside PRE_SIMULATION/POST_SIMULATION jobs.
    """
    widget.add_event(
        make_workflow_event(iteration=0, hook="POST_SIMULATION", job_name="sim_job")
    )
    widget.add_event(
        make_workflow_event(
            iteration=0, hook="POST_EXPERIMENT", job_name="teardown_job"
        )
    )

    labels = [
        widget._iteration_selector.itemText(i)
        for i in range(widget._iteration_selector.count())
    ]
    assert labels == [NO_ITERATION_LABEL, "Iteration 0"]

    widget._iteration_selector.setCurrentIndex(0)
    assert column_values(widget, 2) == ["teardown_job"]

    widget._iteration_selector.setCurrentIndex(1)
    assert column_values(widget, 2) == ["sim_job"]


def test_that_selecting_row_shows_its_stdout_and_stderr(widget):
    widget.add_event(
        make_workflow_event(job_name="first", stdout="out 1", stderr="err 1")
    )
    widget.add_event(
        make_workflow_event(
            job_name="second", job_index=1, stdout="out 2", stderr="err 2"
        )
    )

    widget._table.selectRow(1)

    assert widget._stdout_view.toPlainText() == "out 2"
    assert widget._stderr_view.toPlainText() == "err 2"


def test_that_multiline_output_is_shown_in_full(widget):
    output = "line one\nline two\nline three"
    widget.add_event(make_workflow_event(stdout=output))

    widget._table.selectRow(0)

    assert widget._stdout_view.toPlainText() == output


def test_that_failed_job_is_marked_failed(widget):
    widget.add_event(
        make_workflow_event(job_name="ok", status=WorkflowJobStatus.SUCCESS)
    )
    widget.add_event(
        make_workflow_event(
            job_name="broken", job_index=1, status=WorkflowJobStatus.FAILED
        )
    )

    assert column_values(widget, 3) == ["Succeeded", "Failed"]
    assert widget._table.item(0, 3).background().color() != (
        widget._table.item(1, 3).background().color()
    )


def test_that_cancelled_job_is_marked_cancelled_rather_than_succeeded(widget):
    widget.add_event(
        make_workflow_event(job_name="ok", status=WorkflowJobStatus.SUCCESS)
    )
    widget.add_event(
        make_workflow_event(
            job_name="stopped", job_index=1, status=WorkflowJobStatus.CANCELLED
        )
    )

    assert column_values(widget, 3) == ["Succeeded", "Cancelled"]
    assert widget._table.item(0, 3).background().color() != (
        widget._table.item(1, 3).background().color()
    )


def test_that_job_without_output_shows_no_output_placeholder(widget):
    widget.add_event(make_workflow_event(stdout="", stderr=""))

    widget._table.selectRow(0)

    assert widget._stdout_view.toPlainText() == NO_OUTPUT_PLACEHOLDER
    assert widget._stderr_view.toPlainText() == NO_OUTPUT_PLACEHOLDER


def test_that_switching_iteration_clears_previously_shown_output(widget):
    widget.add_event(make_workflow_event(iteration=0, stdout="iteration zero output"))
    widget.add_event(make_workflow_event(iteration=1, stdout="iteration one output"))
    pick_iteration(widget, 0)
    widget._table.selectRow(0)
    assert widget._stdout_view.toPlainText() == "iteration zero output"

    pick_iteration(widget, 1)

    assert widget._stdout_view.toPlainText() == NO_OUTPUT_PLACEHOLDER


def test_that_loading_events_replaces_previous_rows_with_one_row_per_event(widget):
    widget.load_events([make_workflow_event(job_name="FROM_FIRST_LOAD", iteration=0)])

    widget.load_events(
        [
            make_workflow_event(job_name="FIRST", iteration=0),
            make_workflow_event(job_name="SECOND", iteration=0),
        ]
    )

    assert column_values(widget, 2) == ["FIRST", "SECOND"]


def test_that_loaded_experiment_wide_events_are_grouped_apart_from_iterations(widget):
    widget.load_events(
        [
            make_workflow_event(
                hook="PRE_EXPERIMENT", job_name="BEFORE", iteration=None
            ),
            make_workflow_event(hook="POST_SIMULATION", job_name="DURING", iteration=0),
        ]
    )

    labels = [
        widget._iteration_selector.itemText(index)
        for index in range(widget._iteration_selector.count())
    ]
    assert labels == [NO_ITERATION_LABEL, "Iteration 0"]


def test_that_loading_events_selects_newest_iteration(widget):
    widget.load_events(
        [
            make_workflow_event(job_name="OLDER", iteration=0),
            make_workflow_event(job_name="NEWER", iteration=1),
        ]
    )

    assert widget._iteration_selector.currentText() == "Iteration 1"
    assert column_values(widget, 2) == ["NEWER"]


def test_that_loading_no_events_leaves_empty_table(widget):
    widget.load_events([make_workflow_event(iteration=0)])

    widget.load_events([])

    assert widget._table.rowCount() == 0
    assert widget._iteration_selector.count() == 0
