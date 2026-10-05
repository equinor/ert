import pytest

from ert.gui.experiments.view import WorkflowLogView
from ert.gui.experiments.view import workflow_log as workflow_log_module
from ert.gui.experiments.view.workflow_log import (
    NO_WORKFLOWS_PLACEHOLDER,
    UNREADABLE_WORKFLOWS_PLACEHOLDER,
)
from tests.ert.unit_tests.gui.experiments.conftest import make_workflow_event


@pytest.fixture
def view(qtbot):
    view = WorkflowLogView()
    qtbot.addWidget(view)
    return view


def write_events(path, events):
    path.write_text(
        "".join(f"{event.model_dump_json()}\n" for event in events), encoding="utf-8"
    )


def job_names(view: WorkflowLogView) -> list[str]:
    table = view._workflow_log._table
    return [table.item(row, 2).text() for row in range(table.rowCount())]


def test_that_stored_workflow_events_are_shown_in_workflow_table(view, tmp_path):
    path = tmp_path / "workflow_events.jsonl"
    write_events(path, [make_workflow_event(job_name="STORED_JOB")])

    view.load_events(path)

    assert view._stack.currentWidget() is view._workflow_log
    assert job_names(view) == ["STORED_JOB"]


def test_that_experiment_without_events_shows_placeholder_after_one_that_had_events(
    view, tmp_path
):
    with_events = tmp_path / "with_events.jsonl"
    write_events(with_events, [make_workflow_event()])
    view.load_events(with_events)

    view.load_events(tmp_path / "missing.jsonl")

    assert view._stack.currentWidget() is view._placeholder
    assert view._placeholder.text() == NO_WORKFLOWS_PLACEHOLDER
    assert job_names(view) == []


def test_that_selecting_another_experiment_shows_its_events(view, tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_events(first, [make_workflow_event(job_name="FROM_FIRST")])
    write_events(second, [make_workflow_event(job_name="FROM_SECOND")])

    view.load_events(first)
    view.load_events(second)

    assert job_names(view) == ["FROM_SECOND"]


def test_that_reloading_same_path_shows_events_written_since_last_load(view, tmp_path):
    path = tmp_path / "workflow_events.jsonl"
    view.load_events(path)

    write_events(path, [make_workflow_event(job_name="WRITTEN_LATER")])
    view.load_events(path)

    assert view._stack.currentWidget() is view._workflow_log
    assert job_names(view) == ["WRITTEN_LATER"]


def test_that_unreadable_workflow_event_file_shows_unreadable_placeholder(
    view, tmp_path, monkeypatch
):
    def raise_os_error(path):
        raise OSError("permission denied")

    monkeypatch.setattr(workflow_log_module, "load_workflow_events", raise_os_error)

    view.load_events(tmp_path / "workflow_events.jsonl")

    assert view._stack.currentWidget() is view._placeholder
    assert view._placeholder.text() == UNREADABLE_WORKFLOWS_PLACEHOLDER
