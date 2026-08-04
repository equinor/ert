import pytest

from ert.storage import open_storage
from ert.storage.mode import ModeError


def test_that_appended_workflow_log_entries_accumulate_in_one_file(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(name="exp")

        experiment.append_workflow_log(["first entry\n"])
        experiment.append_workflow_log(["second entry\n", "third entry\n"])

        assert experiment.workflow_log_path.read_text(encoding="utf-8") == (
            "first entry\nsecond entry\nthird entry\n"
        )


def test_that_the_workflow_log_lives_next_to_the_experiment_it_belongs_to(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        first = storage.create_experiment(name="exp")
        second = storage.create_experiment(name="exp")

        first.append_workflow_log(["belongs to the first\n"])
        second.append_workflow_log(["belongs to the second\n"])

        assert first.workflow_log_path.parent == first._path
        assert first.workflow_log_path != second.workflow_log_path
        assert (
            first.workflow_log_path.read_text(encoding="utf-8")
            == "belongs to the first\n"
        )


def test_that_the_workflow_log_keeps_output_that_is_not_ascii(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(name="exp")

        experiment.append_workflow_log(["hei på deg ⏱\n"])

        assert (
            experiment.workflow_log_path.read_text(encoding="utf-8") == "hei på deg ⏱\n"
        )


def test_that_a_workflow_log_cannot_be_appended_to_through_read_only_storage(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(name="exp")
        storage.create_ensemble(experiment, ensemble_size=1, name="ens")
        experiment_id = experiment.id

    with open_storage(tmp_path, mode="r") as storage:
        experiment = storage.get_experiment(experiment_id)

        with pytest.raises(ModeError):
            experiment.append_workflow_log(["should not be written\n"])

        assert not experiment.workflow_log_path.exists()
