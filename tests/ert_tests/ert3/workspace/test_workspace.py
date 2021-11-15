import copy
from pathlib import Path

import pytest

import ert3
import ert

import yaml

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


def test_workspace__validate_ensemble_size(
    workspace, ensemble, stages_config, stages_config_list, base_ensemble_dict
):
    ert3.workspace._workspace._validate_ensemble_size(
        ert3.config.ExperimentConfig(type="evaluation"), ensemble
    )

    ensemble_dict = copy.deepcopy(base_ensemble_dict)
    ensemble_dict["size"] = None
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="An ensemble size must be specified.",
    ):
        ert3.workspace._workspace._validate_ensemble_size(
            ert3.config.ExperimentConfig(type="evaluation"),
            ert3.config.EnsembleConfig.parse_obj(ensemble_dict),
        )

    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="No ensemble size should be specified for a sensitivity analysis.",
    ):
        ert3.workspace._workspace._validate_ensemble_size(
            ert3.config.ExperimentConfig(type="sensitivity", algorithm="one-at-a-time"),
            ensemble,
        )

    ensemble_dict = copy.deepcopy(base_ensemble_dict)
    ensemble_dict["size"] = None
    ert3.workspace._workspace._validate_ensemble_size(
        ert3.config.ExperimentConfig(type="sensitivity", algorithm="one-at-a-time"),
        ert3.config.EnsembleConfig.parse_obj(ensemble_dict),
    )


def test_workspace__validate_inputs(
    workspace, ensemble, stages_config, double_stages_config, base_ensemble_dict
):
    ert3.workspace._workspace._validate_inputs(stages_config, ensemble)

    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="Ensemble and stage inputs do not match.",
    ):
        ert3.workspace._workspace._validate_inputs(double_stages_config, ensemble)

    ensemble_dict = copy.deepcopy(base_ensemble_dict)
    ensemble_dict["forward_model"]["stage"] = "foo"
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="Invalid stage: 'foo'.",
    ):
        ert3.workspace._workspace._validate_inputs(
            double_stages_config,
            ert3.config.EnsembleConfig.parse_obj(ensemble_dict),
        )


def test_workspace_load_experiment_config_validation(
    workspace,
    stages_config,
    base_ensemble_dict,
    stages_config_list,
    double_stages_config_list,
):
    experiments_dir = Path(workspace._path) / _EXPERIMENTS_BASE
    (experiments_dir / "test").mkdir(parents=True)
    with open(experiments_dir / "test" / "ensemble.yml", "w") as f:
        yaml.dump(base_ensemble_dict, f)
    with open(experiments_dir / "test" / "experiment.yml", "w") as f:
        yaml.dump({"type": "evaluation"}, f)
    with open(workspace._path / "stages.yml", "w") as f:
        yaml.dump(stages_config_list, f)
    workspace.load_experiment_config("test")

    ensemble_dict = copy.deepcopy(base_ensemble_dict)
    ensemble_dict["size"] = None
    with open(experiments_dir / "test" / "ensemble.yml", "w") as f:
        yaml.dump(ensemble_dict, f)
    with open(experiments_dir / "test" / "experiment.yml", "w") as f:
        yaml.dump({"type": "sensitivity", "algorithm": "one-at-a-time"}, f)
    with open(workspace._path / "stages.yml", "w") as f:
        yaml.dump(stages_config_list, f)
    workspace.load_experiment_config("test")


def test_workspace_load_experiment_config_size_validation(
    workspace,
    stages_config,
    base_ensemble_dict,
    stages_config_list,
    double_stages_config_list,
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
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="An ensemble size must be specified.",
    ):
        workspace.load_experiment_config("test")

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
        workspace.load_experiment_config("test")


def test_workspace_load_experiment_config_input_validation(
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
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="Ensemble and stage inputs do not match.",
    ):
        workspace.load_experiment_config("test")

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
        match="Invalid stage: 'foo'.",
    ):
        workspace.load_experiment_config("test")
