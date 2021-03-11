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


def test_cli_init(tmpdir):
    with tmpdir.as_cwd():
        args = ["ert3", "init"]
        with patch.object(sys, "argv", args):
            ert3.console.main()


def test_cli_init_twice(tmpdir):
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


def test_cli_init_subfolder(workspace):
    workspace.mkdir("sub_folder").chdir()
    args = ["ert3", "init"]
    with patch.object(sys, "argv", args):
        with pytest.raises(
            ert3.exceptions.IllegalWorkspaceOperation,
            match="Already inside an ERT workspace",
        ):
            ert3.console._console._main()


def test_cli_run_invalid_experiment(workspace):
    args = ["ert3", "run", "this-is-not-an-experiment"]
    with patch.object(sys, "argv", args):
        with pytest.raises(
            ert3.exceptions.IllegalWorkspaceOperation,
            match="this-is-not-an-experiment is not an experiment",
        ):
            ert3.console._console._main()


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


def test_cli_status_no_runs(workspace, capsys):
    experiments_folder = workspace.mkdir(ert3.workspace.EXPERIMENTS_BASE)
    experiments_folder.mkdir("E0")
    experiments = ert3.workspace.get_experiment_names(workspace)

    args = ["ert3", "status"]
    with patch.object(sys, "argv", args):
        ert3.console.main()

    _assert_done_or_pending(capsys.readouterr(), experiments, [])


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


def test_cli_status_no_experiments(workspace, capsys):
    workspace.mkdir(ert3.workspace.EXPERIMENTS_BASE)

    args = ["ert3", "status"]
    with patch.object(sys, "argv", args):
        ert3.console.main()

    captured = capsys.readouterr()
    assert captured.out.strip() == "No experiments present in this workspace"


def test_cli_status_no_experiments_root(workspace):
    args = ["ert3", "status"]
    with patch.object(sys, "argv", args):
        with pytest.raises(
            ert3.exceptions.IllegalWorkspaceState,
            match=f"the workspace {workspace} cannot access experiments",
        ):
            ert3.console._console._main()
