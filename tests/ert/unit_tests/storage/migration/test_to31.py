import json
import logging

import numpy as np

from ert.storage.migration.to31 import migrate


def test_that_migration_removes_fields_not_in_observation_models(tmp_path):
    root = tmp_path / "project"
    root.mkdir()

    exp_path = root / "experiments" / "exp1"
    exp_path.mkdir(parents=True)

    index_data = {
        "id": "exp-id",
        "name": "exp1",
        "ensembles": [],
        "experiment": {
            "experiment_type": "Ensemble Experiment",
            "observations": [
                {
                    "type": "summary_observation",
                    "name": "FOPR_1",
                    "value": 1.0,
                    "error": 0.1,
                    "key": "FOPR",
                    "date": "2024-01-15",
                    "shape_id": None,
                    "some_future_unknown_field": {"foo": "bar"},
                    "std": 0.5,
                },
                {
                    "type": "general_observation",
                    "name": "GEN_1",
                    "data": "POLY_RES",
                    "value": 2.0,
                    "error": 0.2,
                    "restart": 0,
                    "index": 3,
                    "shape_id": None,
                    "response_key": "POLY_RES",
                    "report_step": 0,
                },
                {
                    "type": "rft_observation",
                    "name": "RFT_1",
                    "well": "OP_1",
                    "date": "2024-01-15",
                    "property": "PRESSURE",
                    "value": 250.0,
                    "error": 5.0,
                    "east": 123.5,
                    "north": 456.7,
                    "tvd": 2500.0,
                    "md": None,
                    "shape_id": None,
                    "zone": None,
                    "radius": 3000.0,
                },
                {
                    "type": "breakthrough",
                    "name": "BT_1",
                    "key": "FWPR",
                    "date": "2024-01-15T00:00:00",
                    "error": 0.1,
                    "threshold": 0.5,
                    "shape_id": None,
                    "obsolete_key": "remove me",
                },
            ],
        },
    }
    (exp_path / "index.json").write_text(json.dumps(index_data), encoding="utf-8")

    migrate(root)

    updated = json.loads((exp_path / "index.json").read_text(encoding="utf-8"))
    observations = updated["experiment"]["observations"]

    summary, general, rft, breakthrough = observations

    assert "some_future_unknown_field" not in summary
    assert "std" not in summary
    assert summary["key"] == "FOPR"
    assert np.isclose(summary["value"], 1.0)

    assert "response_key" not in general
    assert "report_step" not in general
    assert general["data"] == "POLY_RES"
    assert general["index"] == 3

    assert "radius" not in rft
    assert rft["well"] == "OP_1"
    assert np.isclose(rft["east"], 123.5)

    assert "obsolete_key" not in breakthrough
    assert np.isclose(breakthrough["threshold"], 0.5)


def test_that_migration_leaves_observations_with_only_known_fields_untouched(tmp_path):
    root = tmp_path / "project"
    root.mkdir()

    exp_path = root / "experiments" / "exp1"
    exp_path.mkdir(parents=True)

    index_data = {
        "id": "exp-id",
        "name": "exp1",
        "ensembles": [],
        "experiment": {
            "experiment_type": "Ensemble Experiment",
            "observations": [
                {
                    "type": "summary_observation",
                    "name": "FOPR_1",
                    "value": 1.0,
                    "error": 0.1,
                    "key": "FOPR",
                    "date": "2024-01-15",
                    "shape_id": None,
                },
            ],
        },
    }
    (exp_path / "index.json").write_text(json.dumps(index_data), encoding="utf-8")

    migrate(root)

    updated = json.loads((exp_path / "index.json").read_text(encoding="utf-8"))
    assert updated == index_data


def test_that_migration_handles_experiments_without_observations(tmp_path):
    root = tmp_path / "project"
    root.mkdir()

    exp_path = root / "experiments" / "exp1"
    exp_path.mkdir(parents=True)

    index_data = {
        "id": "exp-id",
        "name": "exp1",
        "ensembles": [],
        "experiment": {
            "experiment_type": "Ensemble Experiment",
            "parameter_configuration": [{"type": "gen_kw", "name": "PARAM1"}],
        },
    }
    (exp_path / "index.json").write_text(json.dumps(index_data), encoding="utf-8")

    migrate(root)

    updated = json.loads((exp_path / "index.json").read_text(encoding="utf-8"))
    assert updated == index_data


def test_that_migration_logs_the_stripped_keys(tmp_path, caplog):
    root = tmp_path / "project"
    root.mkdir()

    exp_path = root / "experiments" / "exp1"
    exp_path.mkdir(parents=True)

    index_data = {
        "id": "exp-id",
        "name": "exp1",
        "ensembles": [],
        "experiment": {
            "experiment_type": "Ensemble Experiment",
            "observations": [
                {
                    "type": "summary_observation",
                    "name": "FOPR_1",
                    "value": 1.0,
                    "error": 0.1,
                    "key": "FOPR",
                    "date": "2024-01-15",
                    "std": 0.5,
                    "response_key": "FOPR",
                },
            ],
        },
    }
    (exp_path / "index.json").write_text(json.dumps(index_data), encoding="utf-8")

    with caplog.at_level(logging.INFO):
        migrate(root)

    assert "std" in caplog.text
    assert "response_key" in caplog.text
    assert str(exp_path / "index.json") in caplog.text
