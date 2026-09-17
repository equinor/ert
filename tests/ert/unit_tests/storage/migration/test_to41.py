import json

from ert.storage.migration.to40 import migrate


def migrate_and_load_updated_experiment(tmp_path, response_configuration):
    root = tmp_path / "project"
    exp_path = root / "experiments" / "exp1"
    exp_path.mkdir(parents=True)

    index_data = {
        "id": "exp-id",
        "name": "exp1",
        "ensembles": [],
        "experiment": {"response_configuration": response_configuration},
    }
    (exp_path / "index.json").write_text(json.dumps(index_data), encoding="utf-8")

    migrate(root)

    updated = json.loads((exp_path / "index.json").read_text(encoding="utf-8"))
    return updated["experiment"]["response_configuration"]


def test_that_objective_offsets_are_zero_for_every_objective(tmp_path):
    migrated = migrate_and_load_updated_experiment(
        tmp_path,
        [
            {
                "type": "everest_objectives",
                "keys": ["npv", "rf"],
                "input_files": ["npv", "rf"],
                "scales": [1.0, 2.0],
                "weights": [1.0, 1.0],
                "objective_types": ["mean", "mean"],
            }
        ],
    )

    assert migrated[0]["offsets"] == [0.0, 0.0]


def test_that_a_non_objective_response_is_left_alone(tmp_path):
    constraints = {
        "type": "everest_constraints",
        "keys": ["c1"],
        "input_files": ["c1"],
        "scales": [1.0],
        "targets": [None],
        "upper_bounds": [1.0],
        "lower_bounds": [None],
    }

    migrated = migrate_and_load_updated_experiment(tmp_path, [constraints])

    assert migrated == [constraints]
