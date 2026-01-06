import json
from datetime import datetime

import polars as pl

from ert.storage.migration.to25 import migrate


def test_that_migration_creates_experiment_index_with_params_responses_and_obs_configs(
    tmp_path,
):
    root = tmp_path / "project"
    root.mkdir()

    exp_path = root / "experiments" / "exp1"
    obs_path = exp_path / "observations"
    obs_path.mkdir(parents=True)

    (exp_path / "metadata.json").write_text(
        json.dumps({"weights": [0.2, 0.8]}), encoding="utf-8"
    )

    (exp_path / "index.json").write_text(
        json.dumps({"id": "test-id", "name": "test-experiment", "ensembles": []}),
        encoding="utf-8",
    )

    responses = {
        "summary": {
            "type": "summary",
            "name": "summary",
            "keys": ["FOPR", "FGPR", "OVERLAP"],
        },
        "gen_data": {
            "type": "gen_data",
            "name": "gen_data",
            "keys": ["SOME", "OTHER", "OVERLAP"],
        },
    }
    parameters = {
        "P1": {
            "type": "gen_kw",
            "name": "P1",
            "distribution": {"name": "uniform", "min": 0, "max": 1},
        }
    }

    (exp_path / "responses.json").write_text(json.dumps(responses), encoding="utf-8")
    (exp_path / "parameter.json").write_text(json.dumps(parameters), encoding="utf-8")

    gen_df = pl.from_dicts(
        [
            {
                "response_key": "SOME",
                "observation_key": "OBS1",
                "report_step": 1,
                "index": 0,
                "observations": 1.23,
                "std": 0.1,
            },
            {
                "response_key": "SOME",
                "observation_key": "OBS2",
                "report_step": 2,
                "index": 1,
                "observations": 2.34,
                "std": 0.2,
            },
        ]
    )

    summary_df = pl.from_dicts(
        [
            {
                "response_key": "FOPR",
                "observation_key": "FOPR",
                "time": datetime(2020, 1, 1),
                "observations": 3.21,
                "std": 0.05,
            }
        ]
    )

    gen_df.write_parquet(obs_path / "gen_data")
    summary_df.write_parquet(obs_path / "summary")

    migrate(root)

    updated_index = json.loads((exp_path / "index.json").read_text(encoding="utf-8"))

    expected_experiment = {
        "weights": [0.2, 0.8],
        "response_configuration": [
            {
                "type": "summary",
                "name": "summary",
                "keys": ["FOPR", "FGPR", "OVERLAP"],
            },
            {
                "type": "gen_data",
                "name": "gen_data",
                "keys": ["SOME", "OTHER", "OVERLAP"],
            },
        ],
        "parameter_configuration": [
            {
                "type": "gen_kw",
                "name": "P1",
                "distribution": {"name": "uniform", "min": 0, "max": 1},
            }
        ],
        "observations": [
            {
                "type": "general_observation",
                "name": name,
                "data": "SOME",
                "value": value,
                "error": error,
                "restart": restart,
                "index": index,
            }
            for name, value, error, restart, index in [
                ("OBS1", 1.23, 0.1, 1, 0),
                ("OBS2", 2.34, 0.2, 2, 1),
            ]
        ]
        + [
            {
                "type": "summary_observation",
                "name": "FOPR",
                "value": 3.21,
                "error": 0.05,
                "key": "FOPR",
                "date": "2020-01-01",
            },
        ],
    }

    expected_index = {
        "id": "test-id",
        "name": "test-experiment",
        "ensembles": [],
        "experiment": expected_experiment,
    }

    assert updated_index == expected_index


def test_that_experiment_config_migration_handles_empty_responses_json(tmp_path):
    root = tmp_path / "project_no_responses"
    root.mkdir()
    exp = root / "experiments" / "no_responses"
    exp.mkdir(parents=True)
    obs = exp / "observations"
    obs.mkdir()
    (exp / "index.json").write_text(
        json.dumps({"id": "test-id", "name": "test-experiment", "ensembles": []}),
        encoding="utf-8",
    )
    (exp / "responses.json").write_text(json.dumps({}), encoding="utf-8")
    (exp / "parameter.json").write_text(json.dumps({}), encoding="utf-8")
    # metadata.json is assumed to always exist
    (exp / "metadata.json").write_text(json.dumps({}), encoding="utf-8")

    migrate(root)
    updated = json.loads((exp / "index.json").read_text(encoding="utf-8"))
    assert "experiment" in updated
    exp_obj = updated["experiment"]
    assert isinstance(exp_obj.get("response_configuration", []), list)
    assert isinstance(exp_obj.get("parameter_configuration", []), list)


def test_that_experiment_config_migration_handles_empty_parameter_json(tmp_path):
    root = tmp_path / "project_no_parameters"
    root.mkdir()
    exp = root / "experiments" / "no_parameters"
    exp.mkdir(parents=True)
    obs = exp / "observations"
    obs.mkdir()
    (exp / "index.json").write_text(
        json.dumps({"id": "test-id", "name": "test-experiment", "ensembles": []}),
        encoding="utf-8",
    )
    (exp / "responses.json").write_text(
        json.dumps({"summary": {"type": "summary", "name": "summary", "keys": []}}),
        encoding="utf-8",
    )
    (exp / "parameter.json").write_text(json.dumps({}), encoding="utf-8")
    # metadata.json is assumed to always exist
    (exp / "metadata.json").write_text(json.dumps({}), encoding="utf-8")

    migrate(root)
    updated = json.loads((exp / "index.json").read_text(encoding="utf-8"))
    assert "experiment" in updated
    exp_obj = updated["experiment"]
    assert isinstance(exp_obj.get("parameter_configuration", []), list)
    assert isinstance(exp_obj.get("response_configuration", []), list)


def test_that_experiment_config_migration_handles_missing_observations_directory(
    tmp_path,
):
    root = tmp_path / "project_no_obs"
    root.mkdir()
    exp = root / "experiments" / "no_obs"
    exp.mkdir(parents=True)
    (exp / "index.json").write_text(
        json.dumps({"id": "test-id", "name": "test-experiment", "ensembles": []}),
        encoding="utf-8",
    )
    (exp / "responses.json").write_text(json.dumps({}), encoding="utf-8")
    (exp / "parameter.json").write_text(json.dumps({}), encoding="utf-8")
    # metadata.json is assumed to always exist
    (exp / "metadata.json").write_text(json.dumps({}), encoding="utf-8")

    migrate(root)
    updated = json.loads((exp / "index.json").read_text(encoding="utf-8"))
    assert "experiment" in updated
    exp_obj = updated["experiment"]
    assert isinstance(exp_obj.get("observations", []), list)


def test_that_experiment_config_migration_does_not_add_weights_when_missing_in_metadata(
    tmp_path,
):
    root = tmp_path / "project_meta_no_weights"
    root.mkdir()
    exp = root / "experiments" / "meta_no_weights"
    exp.mkdir(parents=True)
    obs = exp / "observations"
    obs.mkdir()
    (exp / "index.json").write_text(
        json.dumps({"id": "test-id", "name": "test-experiment", "ensembles": []}),
        encoding="utf-8",
    )
    (exp / "responses.json").write_text(json.dumps({}), encoding="utf-8")
    (exp / "parameter.json").write_text(json.dumps({}), encoding="utf-8")
    (exp / "metadata.json").write_text(json.dumps({"other": 1}), encoding="utf-8")

    migrate(root)
    updated = json.loads((exp / "index.json").read_text(encoding="utf-8"))
    assert "experiment" in updated
    exp_obj = updated["experiment"]
    assert "weights" not in exp_obj
