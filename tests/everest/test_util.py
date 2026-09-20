import json
from pathlib import Path

import pytest

from ert.storage import open_storage
from ert.storage.local_experiment import ExperimentState, ExperimentStatus
from everest.bin.utils import (
    get_experiment_status,
    remove_show_scaling_warning_setting,
)
from everest.strings import EVEREST
from tests.everest.utils import (
    everest_config_with_defaults,
    relpath,
)

EIGHTCELLS_DATA = relpath(
    "../../test-data/everest/eightcells/eclipse/include/",
    "realizations/realization-0/eclipse/model/EIGHTCELLS.DATA",
)


def test_get_values(change_to_tmpdir):
    exp_dir = Path("the_config_directory")
    exp_file = "the_config_file"
    rel_out_dir = "the_output_directory"
    abs_out_dir = "/the_output_directory"
    exp_dir.mkdir()
    (exp_dir / exp_file).write_text(" ", encoding="utf-8")

    config = everest_config_with_defaults(
        environment={
            "output_folder": abs_out_dir,
            "simulation_folder": "simulation_folder",
        },
        config_path=exp_dir / exp_file,
    )

    config.environment.output_folder = rel_out_dir


def test_get_experiment_status(change_to_tmpdir):
    storage_dir = "."

    # No experiments in storage
    status = get_experiment_status(storage_dir)
    assert status is None

    with open_storage(storage_dir, "w") as writable_storage:
        experiment = writable_storage.create_experiment(name="test_experiment")
        writable_storage.create_ensemble(
            experiment=experiment, name="test_ensemble", ensemble_size=10
        )
        assert len(list(writable_storage.experiments)) == 1
        assert experiment.status is None

    status = get_experiment_status(storage_dir)
    assert status is None

    # Update the experiment status to running
    experiment.status = ExperimentStatus(status=ExperimentState.running)
    status = get_experiment_status(storage_dir)
    assert status.status == ExperimentState.running


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (json.dumps({EVEREST: {"show_scaling_warning": True}}), None),
        (json.dumps({EVEREST: {"show_scaling_warning": False}}), None),
        (
            json.dumps({EVEREST: {"show_scaling_warning": True, "other": 1}}),
            {EVEREST: {"other": 1}},
        ),
        (
            json.dumps({EVEREST: {"show_scaling_warning": True}, "ert": {"foo": 1}}),
            {"ert": {"foo": 1}},
        ),
    ],
)
def test_that_removes_show_scaling_warning_key_or_deletes_empty_file(
    change_to_tmpdir, monkeypatch, content, expected
):
    monkeypatch.setenv("HOME", str(Path.cwd()))
    Path(".ert").write_text(content, encoding="utf-8")

    remove_show_scaling_warning_setting()

    if expected is None:
        assert not Path(".ert").exists()
    else:
        assert json.loads(Path(".ert").read_text(encoding="utf-8")) == expected
