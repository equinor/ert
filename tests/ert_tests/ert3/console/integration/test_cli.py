import copy
import os
import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest
import yaml

import ert
from ert import ert3
from ert_shared.services import Storage

from ....ert_utils import chdir

_EXPERIMENTS_BASE = ert3.workspace._workspace._EXPERIMENTS_BASE


@pytest.mark.parametrize(
    "args",
    [
        ["ert3", "run", "something"],
        ["ert3", "export", "something"],
    ],
)
def test_cli_no_init(tmpdir, args):
    with tmpdir.as_cwd():
        with patch.object(sys, "argv", args):
            with pytest.raises(
                ert.exceptions.IllegalWorkspaceOperation,
                match="Not inside an ERT workspace",
            ):
                ert3.console._console._main()


def test_cli_no_args():
    args = ["ert3"]
    with patch.object(sys, "argv", args):
        ert3.console.main()


@pytest.mark.requires_ert_storage
def test_cli_init(tmpdir, ert_storage):
    with tmpdir.as_cwd():
        args = ["ert3", "init"]
        with patch.object(sys, "argv", args):
            ert3.console.main()


@pytest.mark.requires_ert_storage
def test_cli_init_twice(tmpdir, ert_storage):
    with tmpdir.as_cwd():
        args = ["ert3", "init"]
        with patch.object(sys, "argv", args):
            ert3.console.main()

        with patch.object(sys, "argv", args):
            with pytest.raises(
                ert.exceptions.IllegalWorkspaceOperation,
                match="Already inside an ERT workspace",
            ):
                ert3.console._console._main()


def test_cli_init_without_storage(tmpdir, mocker):
    with tmpdir.as_cwd():
        mocker.patch("ert.storage.init", side_effect=TimeoutError)
        args = ["ert3", "init"]
        with patch.object(sys, "argv", args):
            with pytest.raises(SystemExit, match="Failed to contact storage"):
                ert3.console.main()
        assert not pathlib.Path(".ert").exists()


@pytest.mark.requires_ert_storage
def test_cli_init_subfolder(workspace):
    (workspace._path / "sub_folder").mkdir()
    with chdir(workspace._path / "sub_folder"):
        args = ["ert3", "init"]
        with patch.object(sys, "argv", args):
            with pytest.raises(
                ert.exceptions.IllegalWorkspaceOperation,
                match="Already inside an ERT workspace",
            ):
                ert3.console._console._main()


@pytest.mark.requires_ert_storage
def test_cli_init_invalid_example(tmpdir, ert_storage):
    with tmpdir.as_cwd():
        args = ["ert3", "init", "--example", "something"]
        with patch.object(sys, "argv", args):
            with pytest.raises(
                SystemExit, match="Example something is not a valid ert3 example."
            ):
                ert3.console.main()


@pytest.mark.requires_ert_storage
def test_cli_init_example(tmpdir, ert_storage):
    with tmpdir.as_cwd():
        args = ["ert3", "init", "--example", "polynomial"]
        with patch.object(sys, "argv", args):
            ert3.console.main()
        polynomial_path = pathlib.Path(tmpdir / "polynomial")
        assert polynomial_path.exists() and polynomial_path.is_dir()
        polynomial_files = [file.name for file in polynomial_path.iterdir()]
        assert "stages.yml" in polynomial_files
        assert "parameters.yml" in polynomial_files
        assert "experiments" in polynomial_files


@pytest.mark.requires_ert_storage
def test_cli_init_example_twice(tmpdir, ert_storage):
    with tmpdir.as_cwd():
        args = ["ert3", "init", "--example", "polynomial"]
        with patch.object(sys, "argv", args):
            ert3.console.main()

        with patch.object(sys, "argv", args):
            with pytest.raises(
                SystemExit,
                match="Your working directory already contains example polynomial",
            ):
                ert3.console.main()


@pytest.mark.requires_ert_storage
def test_cli_init_remnant_in_storage(tmpdir, ert_storage):
    """Create a scenario where a workspace has for any reason already been registered
    in storage, but the ".ert" directory is missing or has never been there. This
    could be due to a name conflict, or a user trying to recover from failures."""
    with tmpdir.as_cwd():
        args = ["ert3", "init"]
        with patch.object(sys, "argv", args):
            ert3.console.main()

        pathlib.Path(".ert").rmdir()

        with patch.object(sys, "argv", args):
            with pytest.raises(
                RuntimeError,
                match="already registered in storage",
            ):
                ert3.console.main()


@pytest.mark.requires_ert_storage
def test_cli_init_example_inside_example(tmpdir, ert_storage):
    with tmpdir.as_cwd():
        args = ["ert3", "init", "--example", "polynomial"]
        with patch.object(sys, "argv", args):
            ert3.console.main()

        polynomial_path = tmpdir / "polynomial"
        polynomial_path.chdir()

        with patch.object(sys, "argv", args):
            with pytest.raises(
                SystemExit,
                match="Already inside an ERT workspace",
            ):
                ert3.console.main()


@pytest.mark.requires_ert_storage
def test_cli_run_invalid_experiment(workspace):
    args = ["ert3", "run", "this-is-not-an-experiment"]
    with patch.object(sys, "argv", args):
        with pytest.raises(
            ert.exceptions.IllegalWorkspaceOperation,
            match="this-is-not-an-experiment is not an experiment",
        ):
            ert3.console._console._main()


@pytest.mark.requires_ert_storage
def test_cli_record_load_not_existing_file(workspace):
    args = [
        "ert3",
        "record",
        "load",
        "designed_coefficients",
        str(workspace._path / "doe" / "no_such_file.json"),
    ]
    with patch.object(sys, "argv", args):
        with pytest.raises(SystemExit):
            ert3.console._console._main()


@pytest.mark.requires_ert_storage
def test_cli_record_load(designed_coeffs_record_file):
    args = [
        "ert3",
        "record",
        "load",
        "designed_coefficients",
        str(designed_coeffs_record_file),
    ]
    with patch.object(sys, "argv", args):
        ert3.console._console._main()


@pytest.mark.requires_ert_storage
def test_cli_record_load_supported_mime(designed_coeffs_record_file):
    args = [
        "ert3",
        "record",
        "load",
        "--mime-type",
        "application/json",
        "designed_coefficients",
        str(designed_coeffs_record_file),
    ]
    with patch.object(sys, "argv", args):
        ert3.console._console._main()


def test_cli_record_load_unsupported_mime(capsys):
    args = [
        "ert3",
        "record",
        "load",
        "--mime-type",
        "text/bar.baz",
        "designed_coefficients",
        "foo.bar.baz",
    ]
    with patch.object(sys, "argv", args):
        with pytest.raises(
            SystemExit,
        ):
            ert3.console._console._main()

            captured = capsys.readouterr()
            assert captured.out.strip().contains(
                "error: argument --mime-type: invalid choice: 'text/bar.baz'"
            )


def test_cli_record_load_default_mime(designed_blob_record_file):
    # Remove the .bin extension, forcing the load command to choose a default
    path = pathlib.Path(str(designed_blob_record_file))
    path.rename(path.stem)
    args = [
        "ert3",
        "record",
        "load",
        "designed_coefficients",
        str(path.stem),
    ]
    with patch.object(sys, "argv", args):
        ert3.console._console._main()


def _assert_done_or_pending(captured, experiments, done_indices):
    lines = [
        line.strip()
        for line in captured.out.splitlines()
        if "have run" not in line and "can be run" not in line and line.strip()
    ]

    done = lines[: len(done_indices)]
    pending = lines[len(done_indices) :]

    if done:
        assert "have run" in captured.out
    else:
        assert "have run" not in captured.out

    if pending:
        assert "can be run" in captured.out
    else:
        assert "can be run" not in captured.out

    for idx, experiment in enumerate(experiments):
        if idx in done_indices:
            assert experiment in done
        else:
            assert experiment in pending


@pytest.mark.requires_ert_storage
def test_cli_status_no_runs(workspace, capsys):
    experiments_folder = workspace._path / _EXPERIMENTS_BASE
    (experiments_folder).mkdir(parents=True)
    (experiments_folder / "E0.yml").touch()
    experiments = workspace.get_experiment_names()

    args = ["ert3", "status"]
    with patch.object(sys, "argv", args):
        ert3.console.main()

    _assert_done_or_pending(capsys.readouterr(), experiments, [])


@pytest.mark.requires_ert_storage
def test_cli_status_some_runs(workspace, capsys):
    experiments_folder = workspace._path / _EXPERIMENTS_BASE
    (experiments_folder).mkdir(parents=True)
    (experiments_folder / "E0.yml").touch()
    (experiments_folder / "E1.yml").touch()
    (experiments_folder / "E2.yml").touch()
    (experiments_folder / "E3.yml").touch()
    experiments = list(workspace.get_experiment_names())

    done_indices = [1, 3]
    for idx in done_indices:
        ert.storage.init_experiment(
            experiment_name=experiments[idx],
            parameters={},
            ensemble_size=42,
            responses=[],
        )

    args = ["ert3", "status"]
    with patch.object(sys, "argv", args):
        ert3.console.main()

    _assert_done_or_pending(capsys.readouterr(), experiments, [1, 3])


@pytest.mark.requires_ert_storage
def test_cli_status_all_run(workspace, capsys):
    experiments_folder = workspace._path / _EXPERIMENTS_BASE
    (experiments_folder).mkdir(parents=True)
    (experiments_folder / "E0.yml").touch()

    experiments = workspace.get_experiment_names()

    for experiment in experiments:
        ert.storage.init_experiment(
            experiment_name=experiment,
            parameters={},
            ensemble_size=42,
            responses=[],
        )

    args = ["ert3", "status"]
    with patch.object(sys, "argv", args):
        ert3.console.main()

    _assert_done_or_pending(capsys.readouterr(), experiments, range(len(experiments)))


@pytest.mark.requires_ert_storage
def test_cli_status_no_experiments(workspace, capsys):
    (workspace._path / _EXPERIMENTS_BASE).mkdir()

    args = ["ert3", "status"]
    with patch.object(sys, "argv", args):
        ert3.console.main()

    captured = capsys.readouterr()
    assert captured.out.strip() == "No experiments present in this workspace"


@pytest.mark.requires_ert_storage
def test_cli_status_no_experiments_root(workspace):
    args = ["ert3", "status"]
    with patch.object(sys, "argv", args):
        with pytest.raises(
            ert.exceptions.IllegalWorkspaceState,
            match=f"the workspace {workspace.name} cannot access experiments",
        ):
            ert3.console._console._main()


@pytest.mark.requires_ert_storage
def test_cli_clean_no_runs(workspace):
    experiments_folder = workspace._path / _EXPERIMENTS_BASE
    (experiments_folder).mkdir(parents=True)
    (experiments_folder / "E0.yml").touch()

    assert ert.storage.get_experiment_names(workspace_name=workspace.name) == set()

    args = ["ert3", "status"]
    with patch.object(sys, "argv", args):
        ert3.console.main()

    assert ert.storage.get_experiment_names(workspace_name=workspace.name) == set()


@pytest.mark.requires_ert_storage
def test_cli_clean_all(workspace):
    experiments_folder = workspace._path / _EXPERIMENTS_BASE
    (experiments_folder).mkdir(parents=True)
    experiments = {"E0", " E1"}
    for experiment in experiments:
        (experiments_folder / f"{experiment}.yml").touch()

    experiments = workspace.get_experiment_names()

    for experiment in experiments:
        ert.storage.init_experiment(
            experiment_name=experiment,
            parameters={},
            ensemble_size=42,
            responses=[],
        )

    assert (
        ert.storage.get_experiment_names(workspace_name=workspace.name) == experiments
    )

    args = ["ert3", "clean", "--all"]
    with patch.object(sys, "argv", args):
        ert3.console.main()

    assert ert.storage.get_experiment_names(workspace_name=workspace.name) == set()


@pytest.mark.requires_ert_storage
def test_cli_clean_one(workspace):
    experiments_folder = workspace._path / _EXPERIMENTS_BASE
    (experiments_folder).mkdir(parents=True)
    experiments = {"E0", " E1"}
    for experiment in experiments:
        (experiments_folder / f"{experiment}.yml").touch()
        ert.storage.init_experiment(
            experiment_name=experiment,
            parameters={},
            ensemble_size=42,
            responses=[],
        )

    assert (
        ert.storage.get_experiment_names(workspace_name=workspace.name) == experiments
    )

    deleted_experiment = experiments.pop()

    args = ["ert3", "clean", deleted_experiment]
    with patch.object(sys, "argv", args):
        ert3.console.main()

    assert (
        ert.storage.get_experiment_names(workspace_name=workspace.name) == experiments
    )


@pytest.mark.requires_ert_storage
def test_cli_clean_non_existent_experiment(workspace, capsys):
    experiments_folder = workspace._path / _EXPERIMENTS_BASE
    (experiments_folder).mkdir(parents=True)
    experiments = {"E0", " E1"}
    for experiment in experiments:
        (experiments_folder / f"{experiment}.yml").touch()

    for experiment in experiments:
        ert.storage.init_experiment(
            experiment_name=experiment,
            parameters={},
            ensemble_size=42,
            responses=[],
        )

    assert (
        ert.storage.get_experiment_names(workspace_name=workspace.name) == experiments
    )

    deleted_experiment = experiments.pop()

    args = ["ert3", "clean", deleted_experiment, "non_existent_experiment"]
    with patch.object(sys, "argv", args):
        ert3.console.main()

    assert (
        ert.storage.get_experiment_names(workspace_name=workspace.name) == experiments
    )

    captured = capsys.readouterr()
    assert (
        captured.out.strip() == "Following experiment(s) did not exist:\n"
        "    non_existent_experiment\n"
        "Perhaps you mistyped an experiment name?"
    )


def test_cli_validation_ensemble_function(base_ensemble_dict, capsys, plugin_registry):
    ert3.config.load_ensemble_config(
        base_ensemble_dict, plugin_registry=plugin_registry
    )

    config = copy.deepcopy(base_ensemble_dict)
    config["size"] = "a"
    with pytest.raises(ert.exceptions.ConfigValidationError) as exc_info:
        ert3.config.load_ensemble_config(config, plugin_registry=plugin_registry)
    ert3.console.report_validation_errors(exc_info.value)
    capture = capsys.readouterr()
    assert "Error while loading ensemble configuration data:" in capture.out
    assert "not a valid integer" in capture.out


def test_cli_validation_experiment_function(
    capsys, base_ensemble_dict, plugin_registry
):
    config = copy.deepcopy(base_ensemble_dict)
    config["experiment"]["type"] = []
    with pytest.raises(ert.exceptions.ConfigValidationError) as exc_info:
        ert3.config.load_ensemble_config(config, plugin_registry=plugin_registry)
    ert3.console.report_validation_errors(exc_info.value)
    capture = capsys.readouterr()
    assert "Error while loading ensemble configuration data:" in capture.out
    assert "Unexpected experiment type" in capture.out


@pytest.mark.parametrize(
    "config, expected",
    [
        (
            {"name": "name", "type": "unix", "input": [], "output": []},
            "not a valid tuple",
        ),
        (
            [{"name": "name", "type": "unix", "input": [], "output": []}],
            "field required",
        ),
        (
            [{"name": {}, "type": "unix", "input": [], "output": []}],
            "str type expected",
        ),
    ],
)
def test_cli_validation_stages_function(config, expected, capsys, plugin_registry):
    with pytest.raises(ert.exceptions.ConfigValidationError) as exc_info:
        ert3.config.load_stages_config(config, plugin_registry=plugin_registry)
    ert3.console.report_validation_errors(exc_info.value)
    capture = capsys.readouterr()
    assert "Error while loading stages configuration data:" in capture.out
    assert expected in capture.out


@pytest.mark.requires_ert_storage
def test_cli_validation_ensemble_command(
    base_ensemble_dict, workspace, capsys, plugin_registry
):
    ert3.config.load_ensemble_config(
        base_ensemble_dict, plugin_registry=plugin_registry
    )
    experiments_folder = workspace._path / _EXPERIMENTS_BASE
    (experiments_folder).mkdir(parents=True)
    (experiments_folder / "E0.yml").touch()
    config = copy.deepcopy(base_ensemble_dict)
    config["size"] = "a"
    with open(experiments_folder / "E0.yml", "w") as f:
        yaml.dump(config, f)
    config = []
    with open(workspace._path / "stages.yml", "w") as f:
        yaml.dump(config, f)
    args = ["ert3", "run", "E0"]
    with patch.object(sys, "argv", args):
        with pytest.raises(SystemExit):
            ert3.console.main()
    capture = capsys.readouterr()
    assert "Error while loading ensemble configuration data:" in capture.out
    assert "not a valid integer" in capture.out


@pytest.mark.requires_ert_storage
def test_cli_validation_experiment_command(base_ensemble_dict, workspace, capsys):
    experiments_folder = workspace._path / _EXPERIMENTS_BASE
    (experiments_folder).mkdir(parents=True)
    (experiments_folder / "E0.yml").touch()
    with open(experiments_folder / "E0.yml", "w") as f:
        yaml.dump(base_ensemble_dict, f)
    config = [{"name": "name", "type": "unix", "input": [], "output": []}]
    with open(workspace._path / "stages.yml", "w") as f:
        yaml.dump(config, f)
    args = ["ert3", "run", "E0"]
    with patch.object(sys, "argv", args):
        with pytest.raises(SystemExit):
            ert3.console.main()
    capture = capsys.readouterr()
    assert "Error while loading stages configuration data:" in capture.out


@pytest.mark.requires_ert_storage
def test_cli_validation_stages_command(base_ensemble_dict, workspace, capsys):
    experiments_folder = workspace._path / _EXPERIMENTS_BASE
    (experiments_folder).mkdir(parents=True)
    (experiments_folder / "E0.yml").touch()
    with open(experiments_folder / "E0.yml", "w") as f:
        yaml.dump(base_ensemble_dict, f)
    config = [{"name": {}, "type": "unix", "input": [], "output": []}]
    with open(workspace._path / "stages.yml", "w") as f:
        yaml.dump(config, f)
    args = ["ert3", "run", "E0"]
    with patch.object(sys, "argv", args):
        with pytest.raises(SystemExit):
            ert3.console.main()
    capture = capsys.readouterr()
    assert "Error while loading stages configuration data:" in capture.out
    assert "str type expected" in capture.out


def test_cli_local_test_run(tmpdir):
    with chdir(tmpdir):
        with Storage.start_server():
            args = ["ert3", "init", "--example", "polynomial"]
            with patch.object(sys, "argv", args):
                ert3.console._console._main()

            os.chdir("polynomial")

            # Error testing is performed in this context to reduce
            # test time as a full storage server is involved.
            with pytest.raises(ert.exceptions.ExperimentError):
                with patch.object(
                    sys, "argv", ["ert3", "run", "sensitivity", "--local-test-run"]
                ):
                    ert3.console._console._main()

            with patch.object(
                sys, "argv", ["ert3", "run", "evaluation", "--local-test-run"]
            ):
                ert3.console._console._main()

            experiments = [
                experiment
                for experiment in ert.storage.get_experiment_names(
                    workspace_name="polynomial"
                )
                if experiment.startswith("evaluation-")
            ]
            assert len(experiments) == 1
            run_id = experiments[0].split("-")[1]
            assert pathlib.Path(f"local-test-run-{run_id}/output.json").exists()


def test_cli_local_test_run_specific_realization(tmpdir):
    with chdir(tmpdir):
        with Storage.start_server():
            args = ["ert3", "init", "--example", "polynomial"]
            with patch.object(sys, "argv", args):
                ert3.console._console._main()

            os.chdir("polynomial")

            with pytest.raises(ert.exceptions.ExperimentError):
                with patch.object(
                    sys, "argv", ["ert3", "run", "evaluation", "--realization", "2"]
                ):
                    ert3.console._console._main()

            poly_size = yaml.safe_load(
                pathlib.Path("experiments/evaluation.yml").read_text(encoding="utf-8")
            )["size"]
            not_existing_realization = poly_size  # zero-indexed realizations

            with pytest.raises(
                ert.exceptions.ConfigValidationError,
                match="Realization out of ensemble bounds",
            ):
                assert not_existing_realization > poly_size - 1
                with patch.object(
                    sys,
                    "argv",
                    [
                        "ert3",
                        "run",
                        "evaluation",
                        "--local-test-run",
                        "--realization",
                        str(not_existing_realization),
                    ],
                ):
                    ert3.console._console._main()

            with patch.object(
                sys,
                "argv",
                ["ert3", "run", "evaluation", "--local-test-run", "--realization", "2"],
            ):
                ert3.console._console._main()

            experiments = [
                experiment
                for experiment in ert.storage.get_experiment_names(
                    workspace_name="polynomial"
                )
                if experiment.startswith("evaluation-")
            ]
            assert len(experiments) == 1
            run_id = experiments[0].split("-")[1]
            assert pathlib.Path(f"local-test-run-{run_id}/output.json").exists()


def test_failing_check_service(tmpdir):
    with tmpdir.as_cwd():
        with patch.object(
            sys, "argv", ["ert3", "service", "check", "storage", "--timeout", "1"]
        ):
            with pytest.raises(SystemExit, match="ERROR: Ert storage not found!"):
                ert3.console._console._main()


@pytest.mark.integration_test
def test_check_service(tmpdir, monkeypatch):
    with tmpdir.as_cwd():
        with patch.object(
            sys, "argv", ["ert3", "service", "check", "storage", "--timeout", "10"]
        ):
            with Storage.start_server(timeout=120):
                ert3.console._console._main()


@pytest.mark.parametrize(
    "call_args, expected_call_args",
    [
        (
            ["ert3", "service", "start", "storage"],
            ["ert", "api", "--enable-new-storage"],
        ),
        (
            ["ert3", "service", "start", "storage", "--database-url", "DATABASE_URL"],
            ["ert", "api", "--database-url", "DATABASE_URL"],
        ),
    ],
)
def test_start_service(tmpdir, monkeypatch, call_args, expected_call_args):
    with tmpdir.as_cwd():
        with patch.object(sys, "argv", call_args):
            mock_execvp = MagicMock()
            monkeypatch.setattr(
                "os.execvp",
                mock_execvp,
            )
            ert3.console._console._main()
            mock_execvp.assert_called_once_with("ert", expected_call_args)
