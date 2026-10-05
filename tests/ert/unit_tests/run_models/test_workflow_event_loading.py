import logging

from ert.run_models.event import load_workflow_events
from tests.ert.unit_tests.gui.experiments.conftest import make_workflow_event


def write_events(path, events):
    path.write_text(
        "".join(f"{event.model_dump_json()}\n" for event in events), encoding="utf-8"
    )


def test_that_loaded_workflow_events_keep_write_order_and_captured_output(tmp_path):
    path = tmp_path / "workflow_events.jsonl"
    write_events(
        path,
        [
            make_workflow_event(job_name="FIRST", stdout="on stdout\n", iteration=2),
            make_workflow_event(job_name="SECOND", stderr="on stderr\n"),
        ],
    )

    first, second = load_workflow_events(path)

    assert [first.job_name, second.job_name] == ["FIRST", "SECOND"]
    assert first.stdout == "on stdout\n"
    assert first.iteration == 2
    assert second.stderr == "on stderr\n"


def test_that_experiment_without_workflow_event_file_loads_no_events(tmp_path):
    assert load_workflow_events(tmp_path / "workflow_events.jsonl") == []


def test_that_truncated_final_line_does_not_discard_events_before_it(tmp_path):
    path = tmp_path / "workflow_events.jsonl"
    write_events(path, [make_workflow_event(job_name="COMPLETE")])
    with path.open("a", encoding="utf-8") as fout:
        fout.write('{"job_name": "TRUNCA')

    assert [event.job_name for event in load_workflow_events(path)] == ["COMPLETE"]


def test_that_skipping_unreadable_workflow_event_logs_its_line_number(tmp_path, caplog):
    path = tmp_path / "workflow_events.jsonl"
    write_events(path, [make_workflow_event(job_name="FIRST")])
    with path.open("a", encoding="utf-8") as fout:
        fout.write("not json at all\n")

    with caplog.at_level(logging.WARNING):
        load_workflow_events(path)

    assert "line 2" in caplog.text


def test_that_blank_lines_between_workflow_events_are_not_reported_as_unreadable(
    tmp_path, caplog
):
    path = tmp_path / "workflow_events.jsonl"
    path.write_text(
        f"\n{make_workflow_event().model_dump_json()}\n\n", encoding="utf-8"
    )

    with caplog.at_level(logging.WARNING):
        events = load_workflow_events(path)

    assert len(events) == 1
    assert not caplog.text
