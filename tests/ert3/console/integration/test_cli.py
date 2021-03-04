import ert3

import pytest
import sys
from unittest.mock import patch


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
            workspace=workspace, experiment_name=experiments[idx], parameters=[]
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
            workspace=workspace, experiment_name=experiment, parameters=[]
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
            workspace=workspace, experiment_name=experiment, parameters=[]
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
            workspace=workspace, experiment_name=experiment, parameters=[]
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
            workspace=workspace, experiment_name=experiment, parameters=[]
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
