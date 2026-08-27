import json

import pytest

from ert.storage import open_storage
from ert.storage.mode import ModeError


def test_that_appended_workflow_events_accumulate_as_one_line_each(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(name="exp")

        experiment.append_workflow_events(['{"job": "first"}'])
        experiment.append_workflow_events(['{"job": "second"}', '{"job": "third"}'])

        lines = experiment.workflow_events_path.read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["job"] for line in lines] == [
            "first",
            "second",
            "third",
        ]


def test_that_workflow_events_live_next_to_the_experiment_they_belong_to(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        first = storage.create_experiment(name="exp")
        second = storage.create_experiment(name="exp")

        first.append_workflow_events(['{"job": "belongs to the first"}'])
        second.append_workflow_events(['{"job": "belongs to the second"}'])

        assert first.workflow_events_path.parent == first._path
        assert first.workflow_events_path != second.workflow_events_path
        assert (
            first.workflow_events_path.read_text(encoding="utf-8")
            == '{"job": "belongs to the first"}\n'
        )


def test_that_workflow_events_keep_output_that_is_not_ascii(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(name="exp")

        experiment.append_workflow_events([json.dumps({"stdout": "hei på deg ⏱"})])

        (line,) = experiment.workflow_events_path.read_text(
            encoding="utf-8"
        ).splitlines()
        assert json.loads(line)["stdout"] == "hei på deg ⏱"


def test_that_workflow_events_cannot_be_appended_through_read_only_storage(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(name="exp")
        storage.create_ensemble(experiment, ensemble_size=1, name="ens")
        experiment_id = experiment.id

    with open_storage(tmp_path, mode="r") as storage:
        experiment = storage.get_experiment(experiment_id)

        with pytest.raises(ModeError):
            experiment.append_workflow_events(['{"job": "should not be written"}'])

        assert not experiment.workflow_events_path.exists()
