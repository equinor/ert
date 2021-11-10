from pathlib import Path

import pytest

import ert3
import ert

_EXPERIMENTS_BASE = ert3.workspace._workspace._EXPERIMENTS_BASE


@pytest.mark.requires_ert_storage
def test_workspace_initialize(tmpdir, ert_storage):
    ert3.workspace.initialize(tmpdir)
    ert.storage.init(workspace_name=tmpdir)

    assert (Path(tmpdir) / ert3.workspace._workspace._WORKSPACE_DATA_ROOT).is_dir()

    with pytest.raises(
        ert.exceptions.IllegalWorkspaceOperation,
        match="Already inside an ERT workspace.",
    ):
        ert3.workspace.initialize(tmpdir)
        ert.storage.init(workspace_name=tmpdir)


@pytest.mark.requires_ert_storage
def test_workspace_load(tmpdir, ert_storage):
    with pytest.raises(ert.exceptions.IllegalWorkspaceOperation):
        ert3.workspace.Workspace(tmpdir)
    with pytest.raises(ert.exceptions.IllegalWorkspaceOperation):
        ert3.workspace.Workspace(tmpdir / "foo")
    ert3.workspace.initialize(tmpdir)
    ert.storage.init(workspace_name=tmpdir)
    assert ert3.workspace.Workspace(tmpdir).name == tmpdir
    assert ert3.workspace.Workspace(tmpdir / "foo").name == tmpdir


@pytest.mark.requires_ert_storage
def test_workspace_assert_experiment_exists(tmpdir, ert_storage):
    experiments_dir = Path(tmpdir) / _EXPERIMENTS_BASE

    workspace = ert3.workspace.initialize(tmpdir)
    ert.storage.init(workspace_name=workspace.name)
    Path(experiments_dir / "test1").mkdir(parents=True)

    workspace.assert_experiment_exists("test1")

    with pytest.raises(
        ert.exceptions.IllegalWorkspaceOperation,
        match=f"test2 is not an experiment within the workspace {tmpdir}",
    ):
        workspace.assert_experiment_exists("test2")


@pytest.mark.requires_ert_storage
def test_workspace_get_experiment_names(tmpdir, ert_storage):
    experiments_dir = Path(tmpdir) / _EXPERIMENTS_BASE

    workspace = ert3.workspace.initialize(tmpdir)
    ert.storage.init(workspace_name=workspace.name)
    Path(experiments_dir / "test1").mkdir(parents=True)
    Path(experiments_dir / "test2").mkdir(parents=True)

    assert workspace.get_experiment_names() == {"test1", "test2"}


@pytest.mark.requires_ert_storage
def test_workspace_experiment_has_run(tmpdir, ert_storage):
    experiments_dir = Path(tmpdir) / _EXPERIMENTS_BASE

    workspace = ert3.workspace.initialize(tmpdir)
    ert.storage.init(workspace_name=tmpdir)
    Path(experiments_dir / "test1").mkdir(parents=True)
    Path(experiments_dir / "test2").mkdir(parents=True)

    ert.storage.init_experiment(
        experiment_name="test1",
        parameters={},
        ensemble_size=42,
        responses=[],
    )

    assert "test1" in ert.storage.get_experiment_names(workspace_name=workspace.name)
    assert "test2" not in ert.storage.get_experiment_names(
        workspace_name=workspace.name
    )


@pytest.mark.requires_ert_storage
def test_workspace_export_json(tmpdir, ert_storage):
    experiments_dir = Path(tmpdir) / _EXPERIMENTS_BASE

    workspace = ert3.workspace.initialize(tmpdir)
    Path(experiments_dir / "test1").mkdir(parents=True)

    workspace.export_json("test1", {1: "x", 2: "y"})
    assert (experiments_dir / "test1" / "data.json").exists()

    workspace.export_json("test1", {1: "x", 2: "y"}, output_file="test.json")
    assert (experiments_dir / "test1" / "test.json").exists()
