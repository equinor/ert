import copy
import os
from pathlib import Path

import pytest

import ert3
import ert

import yaml

_EXPERIMENTS_BASE = ert3.workspace._workspace._EXPERIMENTS_BASE
_RESOURCES_BASE = ert3.workspace._workspace._RESOURCES_BASE


@pytest.mark.requires_ert_storage
def test_workspace_initialize(tmpdir, ert_storage):
    ert3.workspace.initialize(tmpdir)
    ert.storage.init(workspace_name=tmpdir)

    data_root = Path(tmpdir) / ert3.workspace._workspace._WORKSPACE_DATA_ROOT
    assert data_root.is_dir()

    with pytest.raises(
        ert.exceptions.IllegalWorkspaceOperation,
        match=f"Already inside an ERT workspace, found {data_root}",
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


def test_workspace__validate_resources(tmpdir, base_ensemble_dict):
    os.chdir(tmpdir)

    workspace = ert3.workspace.initialize(tmpdir)

    ensemble_dict = copy.deepcopy(base_ensemble_dict)
    ensemble_dict["input"] += [
        {"source": "resources.coefficients.json", "record": "coefficients"}
    ]
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="Cannot locate resource: 'coefficients.json'",
    ):
        workspace._validate_resources(
            ert3.config.EnsembleConfig.parse_obj(ensemble_dict),
        )

    resources_dir = Path(tmpdir) / _RESOURCES_BASE

    (resources_dir / "coefficients.json").mkdir(parents=True)
    workspace._validate_resources(ert3.config.EnsembleConfig.parse_obj(ensemble_dict))
    ensemble_dict["input"][-1]["is_directory"] = True
    workspace._validate_resources(ert3.config.EnsembleConfig.parse_obj(ensemble_dict))
    ensemble_dict["input"][-1]["is_directory"] = False
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="Resource must be a regular file: 'coefficients.json'",
    ):
        workspace._validate_resources(
            ert3.config.EnsembleConfig.parse_obj(ensemble_dict)
        )

    (resources_dir / "coefficients.json").rmdir()
    (resources_dir / "coefficients.json").touch()
    ensemble_dict["input"][-1]["is_directory"] = None
    workspace._validate_resources(ert3.config.EnsembleConfig.parse_obj(ensemble_dict))
    ensemble_dict["input"][-1]["is_directory"] = False
    workspace._validate_resources(ert3.config.EnsembleConfig.parse_obj(ensemble_dict))
    ensemble_dict["input"][-1]["is_directory"] = True
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="Resource must be a directory: 'coefficients.json'",
    ):
        workspace._validate_resources(
            ert3.config.EnsembleConfig.parse_obj(ensemble_dict)
        )


def test_workspace_load_experiment_config_validation(
    workspace,
    stages_config,
    base_ensemble_dict,
    stages_config_list,
):
    experiments_dir = Path(workspace._path) / _EXPERIMENTS_BASE
    (experiments_dir / "test").mkdir(parents=True)
    with open(experiments_dir / "test" / "ensemble.yml", "w") as f:
        yaml.dump(base_ensemble_dict, f)
    with open(experiments_dir / "test" / "experiment.yml", "w") as f:
        yaml.dump({"type": "evaluation"}, f)
    with open(workspace._path / "stages.yml", "w") as f:
        yaml.dump(stages_config_list, f)
    with open(workspace._path / "parameters.yml", "w") as f:
        yaml.dump([], f)
    workspace.load_experiment_run_config("test")

    ensemble_dict = copy.deepcopy(base_ensemble_dict)
    ensemble_dict["size"] = None
    with open(experiments_dir / "test" / "ensemble.yml", "w") as f:
        yaml.dump(ensemble_dict, f)
    with open(experiments_dir / "test" / "experiment.yml", "w") as f:
        yaml.dump({"type": "sensitivity", "algorithm": "one-at-a-time"}, f)
    with open(workspace._path / "stages.yml", "w") as f:
        yaml.dump(stages_config_list, f)
    workspace.load_experiment_run_config("test")


def test_workspace_load_experiment_config_size_validation(
    workspace,
    stages_config,
    base_ensemble_dict,
    stages_config_list,
):
    experiments_dir = Path(workspace._path) / _EXPERIMENTS_BASE
    (experiments_dir / "test").mkdir(parents=True)

    ensemble_dict = copy.deepcopy(base_ensemble_dict)
    ensemble_dict["size"] = None
    with open(experiments_dir / "test" / "ensemble.yml", "w") as f:
        yaml.dump(ensemble_dict, f)
    with open(experiments_dir / "test" / "experiment.yml", "w") as f:
        yaml.dump({"type": "evaluation"}, f)
    with open(workspace._path / "stages.yml", "w") as f:
        yaml.dump(stages_config_list, f)
    with open(workspace._path / "parameters.yml", "w") as f:
        yaml.dump([], f)
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="An ensemble size must be specified.",
    ):
        workspace.load_experiment_run_config("test")

    with open(experiments_dir / "test" / "ensemble.yml", "w") as f:
        yaml.dump(base_ensemble_dict, f)
    with open(experiments_dir / "test" / "experiment.yml", "w") as f:
        yaml.dump({"type": "sensitivity", "algorithm": "one-at-a-time"}, f)
    with open(workspace._path / "stages.yml", "w") as f:
        yaml.dump(stages_config_list, f)
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="No ensemble size should be specified for a sensitivity analysis.",
    ):
        workspace.load_experiment_run_config("test")


def test_workspace_load_experiment_config_stages_validation(
    workspace,
    stages_config,
    base_ensemble_dict,
    stages_config_list,
    double_stages_config_list,
):
    experiments_dir = Path(workspace._path) / _EXPERIMENTS_BASE
    (experiments_dir / "test").mkdir(parents=True)
    with open(experiments_dir / "test" / "experiment.yml", "w") as f:
        yaml.dump({"type": "evaluation"}, f)
    with open(experiments_dir / "test" / "ensemble.yml", "w") as f:
        yaml.dump(base_ensemble_dict, f)
    with open(workspace._path / "stages.yml", "w") as f:
        yaml.dump(double_stages_config_list, f)
    with open(workspace._path / "parameters.yml", "w") as f:
        yaml.dump([], f)
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="Ensemble and stage inputs do not match.",
    ):
        workspace.load_experiment_run_config("test")

    ensemble_dict = copy.deepcopy(base_ensemble_dict)
    ensemble_dict["forward_model"]["stage"] = "foo"
    with open(experiments_dir / "test" / "experiment.yml", "w") as f:
        yaml.dump({"type": "evaluation"}, f)
    with open(experiments_dir / "test" / "ensemble.yml", "w") as f:
        yaml.dump(ensemble_dict, f)
    with open(workspace._path / "stages.yml", "w") as f:
        yaml.dump(stages_config_list, f)
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match=(
            "Invalid stage in forward model: 'foo'. "
            "Must be one of: 'evaluate_polynomial'"
        ),
    ):
        workspace.load_experiment_run_config("test")


def test_workspace_load_experiment_config_resources_validation(
    tmpdir,
    base_ensemble_dict,
    stages_config_list,
):
    os.chdir(tmpdir)
    workspace = ert3.workspace.initialize(tmpdir)
    resources_dir = Path(tmpdir) / _RESOURCES_BASE

    script_file = Path("poly.py")
    script_file.touch()

    experiments_dir = Path(workspace._path) / _EXPERIMENTS_BASE
    (experiments_dir / "test").mkdir(parents=True)
    with open(experiments_dir / "test" / "experiment.yml", "w") as f:
        yaml.dump({"type": "evaluation"}, f)
    with open(workspace._path / "stages.yml", "w") as f:
        yaml.dump(stages_config_list, f)
    with open(workspace._path / "parameters.yml", "w") as f:
        yaml.dump([], f)

    ensemble_dict = copy.deepcopy(base_ensemble_dict)
    ensemble_dict["input"] += [
        {"source": "resources.coefficients.json", "record": "coefficients"}
    ]
    with open(experiments_dir / "test" / "ensemble.yml", "w") as f:
        yaml.dump(ensemble_dict, f)
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="Cannot locate resource: 'coefficients.json'",
    ):
        workspace.load_experiment_run_config("test")

    (resources_dir / "coefficients.json").mkdir(parents=True)
    workspace.load_experiment_run_config("test")
    ensemble_dict["input"][-1]["is_directory"] = True
    with open(experiments_dir / "test" / "ensemble.yml", "w") as f:
        yaml.dump(ensemble_dict, f)
    workspace.load_experiment_run_config("test")
    ensemble_dict["input"][-1]["is_directory"] = False
    with open(experiments_dir / "test" / "ensemble.yml", "w") as f:
        yaml.dump(ensemble_dict, f)
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="Resource must be a regular file: 'coefficients.json'",
    ):
        workspace.load_experiment_run_config("test")

    (resources_dir / "coefficients.json").rmdir()
    (resources_dir / "coefficients.json").touch()
    ensemble_dict["input"][-1]["is_directory"] = None
    with open(experiments_dir / "test" / "ensemble.yml", "w") as f:
        yaml.dump(ensemble_dict, f)
    workspace.load_experiment_run_config("test")
    ensemble_dict["input"][-1]["is_directory"] = False
    with open(experiments_dir / "test" / "ensemble.yml", "w") as f:
        yaml.dump(ensemble_dict, f)
    workspace.load_experiment_run_config("test")
    ensemble_dict["input"][-1]["is_directory"] = True
    with open(experiments_dir / "test" / "ensemble.yml", "w") as f:
        yaml.dump(ensemble_dict, f)
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="Resource must be a directory: 'coefficients.json'",
    ):
        workspace.load_experiment_run_config("test")
