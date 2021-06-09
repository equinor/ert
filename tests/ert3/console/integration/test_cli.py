import copy
import pathlib
import sys
from unittest.mock import patch

import pytest
import yaml

import ert3


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
                ert3.exceptions.IllegalWorkspaceOperation,
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
                ert3.exceptions.IllegalWorkspaceOperation,
                match="Already inside an ERT workspace",
            ):
                ert3.console._console._main()


@pytest.mark.requires_ert_storage
def test_cli_init_subfolder(workspace):
    workspace.mkdir("sub_folder").chdir()
    args = ["ert3", "init"]
    with patch.object(sys, "argv", args):
        with pytest.raises(
            ert3.exceptions.IllegalWorkspaceOperation,
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
            ert3.exceptions.IllegalWorkspaceOperation,
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
        str(workspace / "doe" / "no_such_file.json"),
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
    experiments_folder = workspace.mkdir(ert3.workspace.EXPERIMENTS_BASE)
    experiments_folder.mkdir("E0")
    experiments = ert3.workspace.get_experiment_names(workspace)

    args = ["ert3", "status"]
    with patch.object(sys, "argv", args):
        ert3.console.main()

    _assert_done_or_pending(capsys.readouterr(), experiments, [])


@pytest.mark.requires_ert_storage
def test_cli_status_some_runs(workspace, capsys):
    experiments_folder = workspace.mkdir(ert3.workspace.EXPERIMENTS_BASE)
    experiments_folder.mkdir("E0")
    experiments_folder.mkdir("E1")
    experiments_folder.mkdir("E2")
    experiments_folder.mkdir("E3")
    experiments = list(ert3.workspace.get_experiment_names(workspace))

    done_indices = [1, 3]
    for idx in done_indices:
        ert3.storage.init_experiment(
            workspace=workspace,
            experiment_name=experiments[idx],
            parameters=[],
            ensemble_size=42,
        )

    args = ["ert3", "status"]
    with patch.object(sys, "argv", args):
        ert3.console.main()

    _assert_done_or_pending(capsys.readouterr(), experiments, [1, 3])


@pytest.mark.requires_ert_storage
def test_cli_status_all_run(workspace, capsys):
    experiments_folder = workspace.mkdir(ert3.workspace.EXPERIMENTS_BASE)
    experiments_folder.mkdir("E0")

    experiments = ert3.workspace.get_experiment_names(workspace)

    for experiment in experiments:
        ert3.storage.init_experiment(
            workspace=workspace,
            experiment_name=experiment,
            parameters=[],
            ensemble_size=42,
        )

    args = ["ert3", "status"]
    with patch.object(sys, "argv", args):
        ert3.console.main()

    _assert_done_or_pending(capsys.readouterr(), experiments, range(len(experiments)))


@pytest.mark.requires_ert_storage
def test_cli_status_no_experiments(workspace, capsys):
    workspace.mkdir(ert3.workspace.EXPERIMENTS_BASE)

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
            ert3.exceptions.IllegalWorkspaceState,
            match=f"the workspace {workspace} cannot access experiments",
        ):
            ert3.console._console._main()


@pytest.mark.requires_ert_storage
def test_cli_clean_no_runs(workspace):
    experiments_folder = workspace.mkdir(ert3.workspace.EXPERIMENTS_BASE)
    experiments_folder.mkdir("E0")

    assert ert3.storage.get_experiment_names(workspace=workspace) == set()

    args = ["ert3", "status"]
    with patch.object(sys, "argv", args):
        ert3.console.main()

    assert ert3.storage.get_experiment_names(workspace=workspace) == set()


@pytest.mark.requires_ert_storage
def test_cli_clean_all(workspace):
    experiments_folder = workspace.mkdir(ert3.workspace.EXPERIMENTS_BASE)
    experiments = {"E0", " E1"}
    for experiment in experiments:
        experiments_folder.mkdir(experiment)

    experiments = ert3.workspace.get_experiment_names(workspace)

    for experiment in experiments:
        ert3.storage.init_experiment(
            workspace=workspace,
            experiment_name=experiment,
            parameters=[],
            ensemble_size=42,
        )
        ert3.evaluator._evaluator._create_evaluator_tmp_dir(
            workspace, experiment
        ).mkdir(parents=True)
        assert ert3.evaluator._evaluator._create_evaluator_tmp_dir(
            workspace, experiment
        ).exists()

    assert ert3.storage.get_experiment_names(workspace=workspace) == experiments

    args = ["ert3", "clean", "--all"]
    with patch.object(sys, "argv", args):
        ert3.console.main()

    assert ert3.storage.get_experiment_names(workspace=workspace) == set()
    for experiment in experiments:
        assert not ert3.evaluator._evaluator._create_evaluator_tmp_dir(
            workspace, experiment
        ).exists()


@pytest.mark.requires_ert_storage
def test_cli_clean_one(workspace):
    experiments_folder = workspace.mkdir(ert3.workspace.EXPERIMENTS_BASE)
    experiments = {"E0", " E1"}
    for experiment in experiments:
        experiments_folder.mkdir(experiment)
        ert3.storage.init_experiment(
            workspace=workspace,
            experiment_name=experiment,
            parameters=[],
            ensemble_size=42,
        )
        ert3.evaluator._evaluator._create_evaluator_tmp_dir(
            workspace, experiment
        ).mkdir(parents=True)
        assert ert3.evaluator._evaluator._create_evaluator_tmp_dir(
            workspace, experiment
        ).exists()

    assert ert3.storage.get_experiment_names(workspace=workspace) == experiments

    deleted_experiment = experiments.pop()

    args = ["ert3", "clean", deleted_experiment]
    with patch.object(sys, "argv", args):
        ert3.console.main()

    assert ert3.storage.get_experiment_names(workspace=workspace) == experiments
    for experiment in experiments:
        assert ert3.evaluator._evaluator._create_evaluator_tmp_dir(
            workspace, experiment
        ).exists()
    assert not ert3.evaluator._evaluator._create_evaluator_tmp_dir(
        workspace, deleted_experiment
    ).exists()


@pytest.mark.requires_ert_storage
def test_cli_clean_non_existant_experiment(workspace, capsys):
    experiments_folder = workspace.mkdir(ert3.workspace.EXPERIMENTS_BASE)
    experiments = {"E0", " E1"}
    for experiment in experiments:
        experiments_folder.mkdir(experiment)

    for experiment in experiments:
        ert3.storage.init_experiment(
            workspace=workspace,
            experiment_name=experiment,
            parameters=[],
            ensemble_size=42,
        )
        ert3.evaluator._evaluator._create_evaluator_tmp_dir(
            workspace, experiment
        ).mkdir(parents=True)
        assert ert3.evaluator._evaluator._create_evaluator_tmp_dir(
            workspace, experiment
        ).exists()

    assert ert3.storage.get_experiment_names(workspace=workspace) == experiments

    deleted_experiment = experiments.pop()

    args = ["ert3", "clean", deleted_experiment, "non_existant_experiment"]
    with patch.object(sys, "argv", args):
        ert3.console.main()

    assert ert3.storage.get_experiment_names(workspace=workspace) == experiments
    for experiment in experiments:
        assert ert3.evaluator._evaluator._create_evaluator_tmp_dir(
            workspace, experiment
        ).exists()
    assert not ert3.evaluator._evaluator._create_evaluator_tmp_dir(
        workspace, deleted_experiment
    ).exists()

    captured = capsys.readouterr()
    assert (
        captured.out.strip() == "Following experiment(s) did not exist:\n"
        "    non_existant_experiment\n"
        "Perhaps you mistyped an experiment name?"
    )


def test_cli_validation_ensemble_function(base_ensemble_dict, capsys):
    ert3.config.load_ensemble_config(base_ensemble_dict)

    config = copy.deepcopy(base_ensemble_dict)
    config["size"] = "a"
    with pytest.raises(ert3.exceptions.ConfigValidationError) as exc_info:
        ert3.config.load_ensemble_config(config)
    ert3.console.report_validation_errors(exc_info.value)
    capture = capsys.readouterr()
    assert "Error while loading ensemble configuration data:" in capture.out
    assert "not a valid integer" in capture.out


def test_cli_validation_experiment_function(capsys):
    config = {"type": "evaluation"}
    ert3.config.load_experiment_config(config)

    config["type"] = []
    with pytest.raises(ert3.exceptions.ConfigValidationError) as exc_info:
        ert3.config.load_experiment_config(config)
    ert3.console.report_validation_errors(exc_info.value)
    capture = capsys.readouterr()
    assert "Error while loading experiment configuration data:" in capture.out
    assert "unhashable type" in capture.out


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
def test_cli_validation_stages_function(config, expected, capsys):
    with pytest.raises(ert3.exceptions.ConfigValidationError) as exc_info:
        ert3.config.load_stages_config(config)
    ert3.console.report_validation_errors(exc_info.value)
    capture = capsys.readouterr()
    assert "Error while loading stages configuration data:" in capture.out
    assert expected in capture.out


@pytest.mark.requires_ert_storage
def test_cli_validation_ensemble_command(base_ensemble_dict, workspace, capsys):
    ert3.config.load_ensemble_config(base_ensemble_dict)
    experiments_folder = workspace.mkdir(ert3.workspace.EXPERIMENTS_BASE)
    experiments_folder.mkdir("E0")
    config = copy.deepcopy(base_ensemble_dict)
    config["size"] = "a"
    with open(experiments_folder / "E0" / "ensemble.yml", "w") as f:
        yaml.dump(config, f)
    args = ["ert3", "run", "E0"]
    with patch.object(sys, "argv", args):
        ert3.console.main()
    capture = capsys.readouterr()
    assert "Error while loading ensemble configuration data:" in capture.out
    assert "not a valid integer" in capture.out


@pytest.mark.requires_ert_storage
def test_cli_validation_experiment_command(base_ensemble_dict, workspace, capsys):
    experiments_folder = workspace.mkdir(ert3.workspace.EXPERIMENTS_BASE)
    experiments_folder.mkdir("E0")
    with open(experiments_folder / "E0" / "ensemble.yml", "w") as f:
        yaml.dump(base_ensemble_dict, f)
    config = {"type": {}}
    with open(experiments_folder / "E0" / "experiment.yml", "w") as f:
        yaml.dump(config, f)
    config = [{"name": "name", "type": "unix", "input": [], "output": []}]
    with open(workspace / "stages.yml", "w") as f:
        yaml.dump(config, f)
    args = ["ert3", "run", "E0"]
    with patch.object(sys, "argv", args):
        ert3.console.main()
    capture = capsys.readouterr()
    assert "Error while loading stages configuration data:" in capture.out


@pytest.mark.requires_ert_storage
def test_cli_validation_stages_command(base_ensemble_dict, workspace, capsys):
    experiments_folder = workspace.mkdir(ert3.workspace.EXPERIMENTS_BASE)
    experiments_folder.mkdir("E0")
    with open(experiments_folder / "E0" / "ensemble.yml", "w") as f:
        yaml.dump(base_ensemble_dict, f)
    config = {"type": "evaluation"}
    with open(experiments_folder / "E0" / "experiment.yml", "w") as f:
        yaml.dump(config, f)
    config = [{"name": {}, "type": "unix", "input": [], "output": []}]
    with open(workspace / "stages.yml", "w") as f:
        yaml.dump(config, f)
    args = ["ert3", "run", "E0"]
    with patch.object(sys, "argv", args):
        ert3.console.main()
    capture = capsys.readouterr()
    assert "Error while loading stages configuration data:" in capture.out
    assert "str type expected" in capture.out
