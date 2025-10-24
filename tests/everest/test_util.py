import json
import os.path
from pathlib import Path

import pytest

from ert.storage import open_storage
from ert.storage.local_experiment import ExperimentState, ExperimentStatus
from everest import util
from everest.bin.utils import get_experiment_status, show_scaled_controls_warning
from everest.config import EverestConfig
from everest.strings import EVEREST
from tests.everest.utils import (
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

    config = EverestConfig.with_defaults(
        environment={
            "output_folder": abs_out_dir,
            "simulation_folder": "simulation_folder",
        },
        config_path=exp_dir / exp_file,
    )

    config.environment.output_folder = rel_out_dir


def test_makedirs(change_to_tmpdir):
    output_dir = os.path.join("unittest_everest_output")
    cwd = os.getcwd()

    # assert output dir (/tmp/tmpXXXX) is empty
    assert not os.path.isdir(output_dir)
    assert len(os.listdir(cwd)) == 0

    # create output folder
    util.makedirs_if_needed(output_dir)

    # assert output folder created
    assert os.path.isdir(output_dir)
    assert len(os.listdir(cwd)) == 1


def test_makedirs_already_exists(change_to_tmpdir):
    output_dir = os.path.join("unittest_everest_output")
    cwd = os.getcwd()

    # create outputfolder and verify it's existing
    util.makedirs_if_needed(output_dir)
    assert os.path.isdir(output_dir)
    assert len(os.listdir(cwd)) == 1

    # run makedirs_if_needed again, verify nothing happened
    util.makedirs_if_needed(output_dir)
    assert os.path.isdir(output_dir)
    assert len(os.listdir(cwd)) == 1


def test_makedirs_roll_existing(change_to_tmpdir):
    output_dir = os.path.join("unittest_everest_output")
    cwd = os.getcwd()

    # create outputfolder and verify it's existing
    util.makedirs_if_needed(output_dir)
    assert os.path.isdir(output_dir)
    assert len(os.listdir(cwd)) == 1

    # run makedirs_if_needed again, verify old dir rolled
    util.makedirs_if_needed(output_dir, True)
    assert os.path.isdir(output_dir)
    assert len(os.listdir(cwd)) == 2

    # run makedirs_if_needed again, verify old dir rolled
    util.makedirs_if_needed(output_dir, True)
    assert os.path.isdir(output_dir)
    assert len(os.listdir(cwd)) == 3


@pytest.mark.parametrize(
    "user_reply, existing_value, result",
    [
        ("Y", True, False),
        ("y", True, False),
        ("N", True, True),
        ("n", True, True),
        ("NA", False, False),
    ],
)
def test_show_scaled_controls_warning_user_info_file_present(
    change_to_tmpdir, monkeypatch, user_reply, existing_value, result
):
    monkeypatch.setenv("HOME", os.getcwd())
    if existing_value:
        monkeypatch.setattr("builtins.input", lambda _: user_reply)

    user_info = {EVEREST: {"show_scaling_warning": existing_value}}

    with open(".ert", "w", encoding="utf-8") as f:
        json.dump(user_info, f)
    if user_reply.lower() == "n":
        with pytest.raises(SystemExit):
            show_scaled_controls_warning()
    else:
        show_scaled_controls_warning()
    with open(".ert", encoding="utf-8") as f:
        user_info = json.load(f)
    assert user_info.get(EVEREST).get("show_scaling_warning") == result


@pytest.mark.parametrize(
    "user_reply, result",
    [
        ("Y", False),
        ("y", False),
        ("anything else", True),
        ("", True),
        ("N", True),
        ("n", True),
    ],
)
def test_show_scaled_controls_warning_no_user_info_file_present(
    change_to_tmpdir, monkeypatch, user_reply, result
):
    monkeypatch.setenv("HOME", os.getcwd())
    monkeypatch.setattr("builtins.input", lambda _: user_reply)

    assert not Path(".ert").exists()

    if user_reply.lower() == "n":
        with pytest.raises(SystemExit):
            show_scaled_controls_warning()
    else:
        show_scaled_controls_warning()

    assert Path(".ert").exists()

    with open(".ert", encoding="utf-8") as f:
        user_info = json.load(f)

    assert user_info.get(EVEREST), "Expected everest key"
    assert user_info.get(EVEREST).get("show_scaling_warning") == result


@pytest.mark.parametrize(
    "user_reply, result",
    [
        ("Y", False),
        ("y", False),
        ("anything else", None),
        ("", None),
        ("N", None),
        ("n", None),
    ],
)
def test_show_scaled_controls_warning_error_reading_from_user_info(
    change_to_tmpdir, monkeypatch, user_reply, result
):
    monkeypatch.setenv("HOME", os.getcwd())
    monkeypatch.setattr("builtins.input", lambda _: user_reply)

    Path(".ert").write_text("{ not valid json ", encoding="utf-8")

    with (
        pytest.raises(json.decoder.JSONDecodeError),
        open(".ert", encoding="utf-8") as f,
    ):
        json.load(f)

    if user_reply.lower() == "n":
        with pytest.raises(SystemExit):
            show_scaled_controls_warning()
    elif user_reply.lower() == "y":
        show_scaled_controls_warning()

        with open(".ert", encoding="utf-8") as f:
            user_info = json.load(f)
        assert user_info.get(EVEREST).get("show_scaling_warning") == result
    else:
        show_scaled_controls_warning()


@pytest.mark.parametrize(
    "user_reply",
    [
        "Y",
        "y",
        "",
        "N",
        "n",
        "anything else",
    ],
)
def test_show_scaled_controls_warning_error_writing_user_info(
    change_to_tmpdir, monkeypatch, user_reply
):
    monkeypatch.setenv("HOME", os.getcwd())
    monkeypatch.setattr("builtins.input", lambda _: user_reply)

    Path(".ert").touch()
    os.chmod(Path(".ert"), 0o444)
    if user_reply.lower() == "n":
        with pytest.raises(SystemExit):
            show_scaled_controls_warning()
    elif user_reply.lower() == "y":
        show_scaled_controls_warning()


@pytest.mark.parametrize(
    "user_reply, existing_value, result",
    [
        ("Y", True, False),
        ("y", True, False),
        ("N", True, True),
        ("n", True, True),
        ("NA", False, False),
    ],
)
def test_show_scaled_controls_warning_preserves_extra_keys(
    change_to_tmpdir, monkeypatch, user_reply, existing_value, result
):
    monkeypatch.setenv("HOME", os.getcwd())
    if existing_value:
        monkeypatch.setattr("builtins.input", lambda _: user_reply)

    user_info = {
        EVEREST: {"show_scaling_warning": existing_value},
        "ert": {"test_key": 42},
    }

    with open(".ert", "w", encoding="utf-8") as f:
        json.dump(user_info, f)

    if user_reply.lower() == "n":
        with pytest.raises(SystemExit):
            show_scaled_controls_warning()
    else:
        show_scaled_controls_warning()

    with open(".ert", encoding="utf-8") as f:
        user_info = json.load(f)
    assert user_info.get(EVEREST).get("show_scaling_warning") == result
    assert user_info.get("ert").get("test_key") == 42


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
