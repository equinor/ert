import json

from ert.storage.migration.to28 import migrate


def test_that_migration_converts_bool_update_to_strategy_string(tmp_path):
    root = tmp_path / "project"
    root.mkdir()

    exp_path = root / "experiments" / "exp1"
    exp_path.mkdir(parents=True)

    params = [
        {
            "type": "gen_kw",
            "name": "PARAM1",
            "update": True,
            "forward_init": False,
            "distribution": {"name": "uniform"},
        },
        {
            "type": "gen_kw",
            "name": "PARAM2",
            "update": False,
            "forward_init": False,
            "distribution": {"name": "normal"},
        },
        {
            "type": "field",
            "name": "PORO",
            "update": True,
            "forward_init": False,
        },
        {
            "type": "everest_parameters",
            "name": "ctrl",
            "update": False,
            "forward_init": False,
        },
    ]

    index_data = {
        "id": "exp-id",
        "experiment": {"parameter_configuration": params},
    }
    (exp_path / "index.json").write_text(json.dumps(index_data), encoding="utf-8")

    migrate(root)

    updated = json.loads((exp_path / "index.json").read_text(encoding="utf-8"))
    updated_params = updated["experiment"]["parameter_configuration"]

    assert updated_params[0]["update"] == "ADAPTIVE"
    assert updated_params[1]["update"] is None
    assert updated_params[2]["update"] == "ADAPTIVE"
    assert updated_params[3]["update"] is None


def test_that_migration_leaves_string_update_values_unchanged(tmp_path):
    root = tmp_path / "project"
    root.mkdir()

    exp_path = root / "experiments" / "exp1"
    exp_path.mkdir(parents=True)

    params = [
        {
            "type": "gen_kw",
            "name": "PARAM1",
            "update": "ADAPTIVE",
            "forward_init": False,
        },
        {
            "type": "gen_kw",
            "name": "PARAM2",
            "update": None,
            "forward_init": False,
        },
        {
            "type": "gen_kw",
            "name": "PARAM3",
            "update": "DISTANCE",
            "forward_init": False,
        },
    ]

    index_data = {
        "id": "exp-id",
        "experiment": {"parameter_configuration": params},
    }
    (exp_path / "index.json").write_text(json.dumps(index_data), encoding="utf-8")

    migrate(root)

    updated = json.loads((exp_path / "index.json").read_text(encoding="utf-8"))
    updated_params = updated["experiment"]["parameter_configuration"]

    assert updated_params[0]["update"] == "ADAPTIVE"
    assert updated_params[1]["update"] is None
    assert updated_params[2]["update"] == "DISTANCE"


def test_that_migration_handles_missing_experiments_dir(tmp_path):
    root = tmp_path / "project"
    root.mkdir()

    # Should not raise
    migrate(root)
