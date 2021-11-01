from pathlib import Path

import pytest

import ert3
import ert


@pytest.mark.requires_ert_storage
def test_workspace_initialize(tmpdir, ert_storage):
    ert3.workspace.initialize(tmpdir)

    assert (Path(tmpdir) / ert3._WORKSPACE_DATA_ROOT).is_dir()

    with pytest.raises(
        ert.exceptions.IllegalWorkspaceOperation,
        match="Already inside an ERT workspace.",
    ):
        ert3.workspace.initialize(tmpdir)


@pytest.mark.requires_ert_storage
def test_workspace_load(tmpdir, ert_storage):
    assert ert3.workspace.load(tmpdir) is None
    assert ert3.workspace.load(tmpdir / "foo") is None
    ert3.workspace.initialize(tmpdir)
    assert ert3.workspace.load(tmpdir) == tmpdir
    assert ert3.workspace.load(tmpdir / "foo") == tmpdir


@pytest.mark.requires_ert_storage
def test_workspace_assert_experiment_exists(tmpdir, ert_storage):
    experiments_dir = Path(tmpdir) / ert3.workspace.EXPERIMENTS_BASE
    with pytest.raises(
        ert.exceptions.IllegalWorkspaceState,
        match=f"the workspace {tmpdir} cannot access experiments",
    ):
        ert3.workspace.get_experiment_names(tmpdir)

    ert3.workspace.initialize(tmpdir)
    Path(experiments_dir / "test1").mkdir(parents=True)

    ert3.workspace.assert_experiment_exists(tmpdir, "test1")

    with pytest.raises(
        ert.exceptions.IllegalWorkspaceOperation,
        match=f"test2 is not an experiment within the workspace {tmpdir}",
    ):
        ert3.workspace.assert_experiment_exists(tmpdir, "test2")


@pytest.mark.requires_ert_storage
def test_workspace_assert_get_experiment_names(tmpdir, ert_storage):
    experiments_dir = Path(tmpdir) / ert3.workspace.EXPERIMENTS_BASE
    with pytest.raises(
        ert.exceptions.IllegalWorkspaceState,
        match=f"the workspace {tmpdir} cannot access experiments",
    ):
        ert3.workspace.get_experiment_names(tmpdir)

    ert3.workspace.initialize(tmpdir)
    Path(experiments_dir / "test1").mkdir(parents=True)
    Path(experiments_dir / "test2").mkdir(parents=True)

    assert ert3.workspace.get_experiment_names(tmpdir) == {"test1", "test2"}


@pytest.mark.requires_ert_storage
def test_workspace_experiment_has_run(tmpdir, ert_storage):
    experiments_dir = Path(tmpdir) / ert3.workspace.EXPERIMENTS_BASE
    with pytest.raises(
        ert.exceptions.IllegalWorkspaceState,
        match=f"the workspace {tmpdir} cannot access experiments",
    ):
        ert3.workspace.get_experiment_names(tmpdir)

    ert3.workspace.initialize(tmpdir)
    Path(experiments_dir / "test1").mkdir(parents=True)
    Path(experiments_dir / "test2").mkdir(parents=True)

    ert.storage.init_experiment(
        experiment_name="test1",
        parameters={},
        ensemble_size=42,
        responses=[],
    )

    assert ert3.workspace.experiment_has_run(tmpdir, "test1")
    assert not ert3.workspace.experiment_has_run(tmpdir, "test2")


@pytest.mark.requires_ert_storage
def test_workspace_export_json(tmpdir, ert_storage):
    experiments_dir = Path(tmpdir) / ert3.workspace.EXPERIMENTS_BASE

    ert3.workspace.initialize(tmpdir)
    Path(experiments_dir / "test1").mkdir(parents=True)

    ert.storage.init_experiment(
        experiment_name="test1",
        parameters={},
        ensemble_size=42,
        responses=[],
    )

    ert3.workspace.export_json(tmpdir, "test1", {1: "x", 2: "y"})
    assert (experiments_dir / "test1" / "data.json").exists()

    ert3.workspace.export_json(
        tmpdir, "test1", {1: "x", 2: "y"}, output_file="test.json"
    )
    assert (experiments_dir / "test1" / "test.json").exists()
