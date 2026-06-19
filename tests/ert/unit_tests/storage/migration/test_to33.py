import json
import logging

from ert.storage.migration.to33 import migrate


def test_that_migration_removes_fields_not_in_response_config_models(tmp_path):
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
            "response_configuration": [
                {
                    "type": "summary",
                    "input_files": ["CASE"],
                    "keys": ["FOPR"],
                    "has_finalized_keys": True,
                    "legacy_summary_key": "remove me",
                },
                {
                    "type": "gen_data",
                    "input_files": ["gen_%d.out"],
                    "keys": ["GEN_1"],
                    "report_steps_list": [[0, 1]],
                    "has_finalized_keys": True,
                    "report_step": 0,
                },
                {
                    "type": "rft",
                    "name": "rft",
                    "input_files": ["CASE.DATA"],
                    "keys": ["RFT_KEY"],
                    "has_finalized_keys": False,
                    "data_to_read": {"OP_1": {"*": ["PRESSURE"]}},
                    "zonemap": None,
                    "approximate_missing_values": True,
                    "response_key": "RFT_KEY",
                    "locations": [12, 12, 12],
                },
            ],
            "derived_response_configuration": [
                {
                    "type": "breakthrough",
                    "keys": ["BT_1"],
                    "summary_keys": ["FWPR"],
                    "thresholds": [0.5],
                    "observed_dates": ["2024-01-15T00:00:00"],
                    "has_finalized_keys": True,
                    "deprecated": True,
                },
                {
                    "type": "everest_constraints",
                    "input_files": ["constraints.txt"],
                    "keys": ["CONS_1"],
                    "scales": [1.0],
                    "targets": [0.0],
                    "upper_bounds": [1.0],
                    "lower_bounds": [None],
                    "has_finalized_keys": True,
                    "max_submit": 1,
                },
                {
                    "type": "everest_objectives",
                    "input_files": ["objectives.txt"],
                    "keys": ["OBJ_1"],
                    "scales": [1.0],
                    "weights": [1.0],
                    "objective_types": ["mean"],
                    "has_finalized_keys": True,
                    "random_seed": 123,
                },
            ],
        },
    }
    (exp_path / "index.json").write_text(json.dumps(index_data), encoding="utf-8")

    migrate(root)

    updated = json.loads((exp_path / "index.json").read_text(encoding="utf-8"))
    responses = updated["experiment"]["response_configuration"]
    derived_responses = updated["experiment"]["derived_response_configuration"]

    summary, gen_data, rft = responses
    breakthrough, everest_constraints, everest_objectives = derived_responses

    assert "legacy_summary_key" not in summary
    assert summary["keys"] == ["FOPR"]

    assert "report_step" not in gen_data
    assert gen_data["report_steps_list"] == [[0, 1]]

    assert "response_key" not in rft
    assert rft["approximate_missing_values"] is True

    assert "deprecated" not in breakthrough
    assert breakthrough["summary_keys"] == ["FWPR"]

    assert "max_submit" not in everest_constraints
    assert everest_constraints["upper_bounds"] == [1.0]

    assert "random_seed" not in everest_objectives
    assert everest_objectives["objective_types"] == ["mean"]


def test_that_migration_leaves_response_configs_with_only_known_fields_untouched(
    tmp_path,
):
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
            "response_configuration": [
                {
                    "type": "summary",
                    "input_files": ["CASE"],
                    "keys": ["FOPR"],
                    "has_finalized_keys": True,
                },
            ],
            "derived_response_configuration": [
                {
                    "type": "breakthrough",
                    "keys": ["BT_1"],
                    "summary_keys": ["FWPR"],
                    "thresholds": [0.5],
                    "observed_dates": ["2024-01-15T00:00:00"],
                    "has_finalized_keys": True,
                },
            ],
        },
    }
    (exp_path / "index.json").write_text(json.dumps(index_data), encoding="utf-8")

    migrate(root)

    updated = json.loads((exp_path / "index.json").read_text(encoding="utf-8"))
    assert updated == index_data


def test_that_migration_handles_experiments_without_response_configs(tmp_path):
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
            "response_configuration": [
                {
                    "type": "rft",
                    "name": "rft",
                    "input_files": ["CASE.DATA"],
                    "keys": ["RFT_KEY"],
                    "has_finalized_keys": False,
                    "data_to_read": {"OP_1": {"*": ["PRESSURE"]}},
                    "zonemap": None,
                    "approximate_missing_values": True,
                    "response_key": "RFT_KEY",
                    "report_step": 0,
                },
            ],
            "derived_response_configuration": [
                {
                    "type": "breakthrough",
                    "keys": ["BT_1"],
                    "summary_keys": ["FWPR"],
                    "thresholds": [0.5],
                    "observed_dates": ["2024-01-15T00:00:00"],
                    "has_finalized_keys": True,
                    "deprecated": True,
                },
            ],
        },
    }
    (exp_path / "index.json").write_text(json.dumps(index_data), encoding="utf-8")

    with caplog.at_level(logging.INFO):
        migrate(root)

    assert "response_key" in caplog.text
    assert "report_step" in caplog.text
    assert "deprecated" in caplog.text
    assert str(exp_path / "index.json") in caplog.text
