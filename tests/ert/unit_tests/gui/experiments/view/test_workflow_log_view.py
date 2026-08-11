from datetime import UTC, datetime
from uuid import uuid4

import pytest

from ert.gui.experiments.view import WorkflowLogView
from ert.gui.experiments.view import workflow_log as workflow_log_module
from ert.run_models.event import WorkflowEvent
from ert.workflow_runner import WorkflowJobStatus


def make_event(job_name: str = "my_job") -> WorkflowEvent:
    return WorkflowEvent(
        run_id=uuid4(),
        hook="PRE_EXPERIMENT",
        workflow_name="my_workflow",
        job_name=job_name,
        job_index=0,
        arguments=[],
        stdout="hello",
        stderr="",
        status=WorkflowJobStatus.SUCCESS,
        timestamp=datetime(2024, 1, 1, 12, 30, 45, tzinfo=UTC),
        iteration=None,
    )


@pytest.fixture
def view(qtbot):
    view = WorkflowLogView()
    qtbot.addWidget(view)
    return view


def write_events(path, events):
    path.write_text(
        "".join(f"{event.model_dump_json()}\n" for event in events), encoding="utf-8"
    )


def test_that_an_experiment_without_workflow_events_shows_the_placeholder(
    view, tmp_path
):
    view.load_events(tmp_path / "workflow_events.jsonl")

    assert view._stack.currentWidget() is view._placeholder


def test_that_stored_workflow_events_are_shown_in_the_workflow_table(view, tmp_path):
    path = tmp_path / "workflow_events.jsonl"
    write_events(path, [make_event(job_name="STORED_JOB")])

    view.load_events(path)

    assert view._stack.currentWidget() is view._workflow_log
    assert view._workflow_log._table.item(0, 2).text() == "STORED_JOB"


def test_that_workflow_events_are_not_reread_when_the_path_is_unchanged(
    view, tmp_path, monkeypatch
):
    path = tmp_path / "workflow_events.jsonl"
    write_events(path, [make_event()])
    load_call_count = 0

    def counting_load(path):
        nonlocal load_call_count
        load_call_count += 1
        return [make_event()]

    monkeypatch.setattr(workflow_log_module, "load_workflow_events", counting_load)

    view.load_events(path)
    view.load_events(path)

    assert load_call_count == 1


def test_that_selecting_another_experiment_shows_that_experiments_events(
    view, tmp_path
):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_events(first, [make_event(job_name="FROM_FIRST")])
    write_events(second, [make_event(job_name="FROM_SECOND")])

    view.load_events(first)
    view.load_events(second)

    assert view._workflow_log._table.item(0, 2).text() == "FROM_SECOND"
    assert view._workflow_log._table.rowCount() == 1


def test_that_an_experiment_without_events_shows_the_placeholder_after_one_that_had(
    view, tmp_path
):
    with_events = tmp_path / "with_events.jsonl"
    write_events(with_events, [make_event()])
    view.load_events(with_events)

    view.load_events(tmp_path / "missing.jsonl")

    assert view._stack.currentWidget() is view._placeholder
    assert view._workflow_log._table.rowCount() == 0


def test_that_an_unreadable_workflow_event_file_shows_the_placeholder(
    view, tmp_path, monkeypatch
):
    def raise_os_error(path):
        raise OSError("permission denied")

    monkeypatch.setattr(workflow_log_module, "load_workflow_events", raise_os_error)

    view.load_events(tmp_path / "workflow_events.jsonl")

    assert view._stack.currentWidget() is view._placeholder
