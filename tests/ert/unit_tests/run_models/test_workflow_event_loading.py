import logging
from datetime import UTC, datetime
from uuid import uuid4

from ert.run_models.event import RunModelWorkflowLogEvent, load_workflow_log_events


def _event(job_name: str = "JOB", **kwargs) -> RunModelWorkflowLogEvent:
    return RunModelWorkflowLogEvent(
        **{
            "run_id": uuid4(),
            "hook": "PRE_EXPERIMENT",
            "workflow_name": "a_workflow",
            "job_name": job_name,
            "job_index": 0,
            "arguments": [],
            "stdout": "",
            "stderr": "",
            "failed": False,
            "timestamp": datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            **kwargs,
        }
    )


def _write(path, events):
    path.write_text(
        "".join(f"{event.model_dump_json()}\n" for event in events), encoding="utf-8"
    )


def test_that_loading_workflow_events_returns_them_in_the_order_they_were_written(
    tmp_path,
):
    path = tmp_path / "workflow_events.jsonl"
    _write(path, [_event(job_name="FIRST"), _event(job_name="SECOND")])

    assert [event.job_name for event in load_workflow_log_events(path)] == [
        "FIRST",
        "SECOND",
    ]


def test_that_loading_workflow_events_preserves_the_captured_output(tmp_path):
    path = tmp_path / "workflow_events.jsonl"
    _write(path, [_event(stdout="on stdout\n", stderr="on stderr\n", iteration=2)])

    (event,) = load_workflow_log_events(path)

    assert event.stdout == "on stdout\n"
    assert event.stderr == "on stderr\n"
    assert event.iteration == 2


def test_that_an_experiment_without_a_workflow_event_file_loads_no_events(tmp_path):
    assert load_workflow_log_events(tmp_path / "workflow_events.jsonl") == []


def test_that_a_truncated_final_line_does_not_discard_the_events_before_it(tmp_path):
    path = tmp_path / "workflow_events.jsonl"
    _write(path, [_event(job_name="COMPLETE")])
    with path.open("a", encoding="utf-8") as fout:
        fout.write('{"job_name": "TRUNCA')

    assert [event.job_name for event in load_workflow_log_events(path)] == ["COMPLETE"]


def test_that_skipping_an_unreadable_workflow_event_names_the_line_it_was_on(
    tmp_path, caplog
):
    path = tmp_path / "workflow_events.jsonl"
    _write(path, [_event(job_name="FIRST")])
    with path.open("a", encoding="utf-8") as fout:
        fout.write("not json at all\n")

    with caplog.at_level(logging.WARNING):
        load_workflow_log_events(path)

    assert "line 2" in caplog.text


def test_that_blank_lines_between_workflow_events_are_not_reported_as_unreadable(
    tmp_path, caplog
):
    path = tmp_path / "workflow_events.jsonl"
    path.write_text(f"\n{_event().model_dump_json()}\n\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        events = load_workflow_log_events(path)

    assert len(events) == 1
    assert not caplog.text
