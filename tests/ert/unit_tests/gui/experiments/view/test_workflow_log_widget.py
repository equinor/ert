from datetime import UTC, datetime
from uuid import uuid4

import pytest

from ert.gui.experiments.view import WorkflowLogWidget
from ert.gui.experiments.view.workflow_log import (
    NO_ITERATION_LABEL,
    NO_OUTPUT_PLACEHOLDER,
)
from ert.run_models.event import RunModelWorkflowLogEvent

RUN_ID = uuid4()


def make_event(
    *,
    hook: str = "POST_SIMULATION",
    workflow_name: str = "my_workflow",
    job_name: str = "my_job",
    job_index: int = 0,
    arguments: list[str] | None = None,
    stdout: str = "hello",
    stderr: str = "",
    failed: bool = False,
    cancelled: bool = False,
    iteration: int | None = 0,
) -> RunModelWorkflowLogEvent:
    return RunModelWorkflowLogEvent(
        run_id=RUN_ID,
        hook=hook,
        workflow_name=workflow_name,
        job_name=job_name,
        job_index=job_index,
        arguments=arguments or [],
        stdout=stdout,
        stderr=stderr,
        failed=failed,
        cancelled=cancelled,
        timestamp=datetime(2024, 1, 1, 12, 30, 45, tzinfo=UTC),
        iteration=iteration,
    )


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


def select_row(widget: WorkflowLogWidget, row: int) -> None:
    widget._table.selectRow(row)


def pick_iteration(widget: WorkflowLogWidget, index: int) -> None:
    """Select an iteration the way a user would, via the combo box."""
    widget._iteration_selector.setCurrentIndex(index)
    widget._iteration_selector.activated.emit(index)


def test_that_workflow_log_widget_starts_with_no_rows(widget):
    assert widget._table.rowCount() == 0
    assert widget._iteration_selector.count() == 0


def test_that_each_workflow_log_event_adds_one_table_row(widget):
    widget.add_event(make_event(job_name="first"))
    widget.add_event(make_event(job_name="second", job_index=1))

    assert widget._table.rowCount() == 2
    assert column_values(widget, 2) == ["first", "second"]


def test_that_job_arguments_are_shown_next_to_the_job_name(widget):
    widget.add_event(make_event(job_name="echo", arguments=["a", "b"]))

    assert column_values(widget, 2) == ["echo(a, b)"]


def test_that_iteration_selector_lists_every_iteration_seen_in_events(widget):
    widget.add_event(make_event(iteration=1))
    widget.add_event(make_event(iteration=0))
    widget.add_event(make_event(iteration=1))

    labels = [
        widget._iteration_selector.itemText(i)
        for i in range(widget._iteration_selector.count())
    ]
    assert labels == ["Iteration 0", "Iteration 1"]


def test_that_selecting_an_iteration_shows_only_that_iterations_jobs(widget):
    widget.add_event(make_event(iteration=0, job_name="job_in_iter_0"))
    widget.add_event(make_event(iteration=1, job_name="job_in_iter_1"))

    pick_iteration(widget, 0)
    assert column_values(widget, 2) == ["job_in_iter_0"]

    pick_iteration(widget, 1)

    assert column_values(widget, 2) == ["job_in_iter_1"]


def test_that_events_arriving_for_the_selected_iteration_are_appended_live(widget):
    widget.add_event(make_event(iteration=0, job_name="first"))
    widget.add_event(make_event(iteration=1, job_name="other"))
    pick_iteration(widget, 1)

    widget.add_event(make_event(iteration=1, job_name="second"))

    assert column_values(widget, 2) == ["other", "second"]


def test_that_a_new_iteration_is_selected_automatically_until_the_user_picks_one(
    widget,
):
    widget.add_event(make_event(iteration=0, job_name="job_in_iter_0"))

    widget.add_event(make_event(iteration=1, job_name="job_in_iter_1"))

    assert widget._iteration_selector.currentText() == "Iteration 1"
    assert column_values(widget, 2) == ["job_in_iter_1"]


def test_that_a_new_iteration_does_not_override_an_iteration_picked_by_the_user(widget):
    widget.add_event(make_event(iteration=0, job_name="job_in_iter_0"))
    pick_iteration(widget, 0)

    widget.add_event(make_event(iteration=1, job_name="job_in_iter_1"))

    assert widget._iteration_selector.currentText() == "Iteration 0"
    assert column_values(widget, 2) == ["job_in_iter_0"]


def test_that_events_without_an_iteration_are_listed_first_in_the_selector(widget):
    widget.add_event(make_event(iteration=1, job_name="job_in_iter_1"))
    widget.add_event(
        make_event(iteration=None, hook="PRE_EXPERIMENT", job_name="early_job")
    )

    assert widget._iteration_selector.itemText(0) == NO_ITERATION_LABEL
    assert widget._iteration_selector.itemData(0) is None


def test_that_selecting_the_pre_post_experiment_entry_shows_jobs_without_iteration(
    widget,
):
    widget.add_event(make_event(iteration=1, job_name="job_in_iter_1"))
    widget.add_event(
        make_event(iteration=None, hook="PRE_EXPERIMENT", job_name="early_job")
    )

    widget._iteration_selector.setCurrentIndex(0)

    assert column_values(widget, 0) == ["PRE_EXPERIMENT"]
    assert column_values(widget, 2) == ["early_job"]


def test_that_post_experiment_events_are_grouped_as_pre_post_experiment_even_with_an_iteration(  # ruff: ignore[line-too-long]
    widget,
):
    """POST_EXPERIMENT hooks run once for the whole experiment, but by then an
    ensemble already exists, so the event carries a real iteration number. It
    should still land under "Pre/post experiment", not under that iteration's
    tab alongside PRE_SIMULATION/POST_SIMULATION jobs.
    """
    widget.add_event(
        make_event(iteration=0, hook="POST_SIMULATION", job_name="sim_job")
    )
    widget.add_event(
        make_event(iteration=0, hook="POST_EXPERIMENT", job_name="teardown_job")
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


def test_that_selecting_a_row_shows_that_jobs_stdout_and_stderr(widget):
    widget.add_event(make_event(job_name="first", stdout="out 1", stderr="err 1"))
    widget.add_event(
        make_event(job_name="second", job_index=1, stdout="out 2", stderr="err 2")
    )

    select_row(widget, 1)

    assert widget._stdout_view.toPlainText() == "out 2"
    assert widget._stderr_view.toPlainText() == "err 2"


def test_that_multiline_output_is_shown_in_full(widget):
    output = "line one\nline two\nline three"
    widget.add_event(make_event(stdout=output))

    select_row(widget, 0)

    assert widget._stdout_view.toPlainText() == output


def test_that_a_failed_job_is_marked_failed(widget):
    widget.add_event(make_event(job_name="ok", failed=False))
    widget.add_event(make_event(job_name="broken", job_index=1, failed=True))

    assert column_values(widget, 3) == ["Succeeded", "Failed"]
    assert widget._table.item(0, 3).background().color() != (
        widget._table.item(1, 3).background().color()
    )


def test_that_a_cancelled_job_is_marked_cancelled_rather_than_succeeded(widget):
    widget.add_event(make_event(job_name="ok", failed=False, cancelled=False))
    widget.add_event(
        make_event(job_name="stopped", job_index=1, failed=False, cancelled=True)
    )

    assert column_values(widget, 3) == ["Succeeded", "Cancelled"]
    assert widget._table.item(0, 3).background().color() != (
        widget._table.item(1, 3).background().color()
    )


def test_that_a_job_with_no_output_shows_a_no_output_placeholder(widget):
    widget.add_event(make_event(stdout="", stderr=""))

    select_row(widget, 0)

    assert widget._stdout_view.toPlainText() == NO_OUTPUT_PLACEHOLDER
    assert widget._stderr_view.toPlainText() == NO_OUTPUT_PLACEHOLDER


def test_that_switching_iteration_clears_the_previously_shown_output(widget):
    widget.add_event(make_event(iteration=0, stdout="iteration zero output"))
    widget.add_event(make_event(iteration=1, stdout="iteration one output"))
    pick_iteration(widget, 0)
    select_row(widget, 0)
    assert widget._stdout_view.toPlainText() == "iteration zero output"

    pick_iteration(widget, 1)

    assert widget._stdout_view.toPlainText() == NO_OUTPUT_PLACEHOLDER
