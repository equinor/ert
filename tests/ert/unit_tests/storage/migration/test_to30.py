import json
import logging

from ert.storage.migration.to30 import migrate


def test_that_migration_removes_fields_not_in_experiment_config(tmp_path):
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
            "storage_path": "/project/storage",
            "runpath_file": "/project/runpath_file",
            "user_config_file": "/project/model.ert",
            "env_vars": {"OMP_NUM_THREADS": "1"},
            "env_pr_fm_step": {"ECLIPSE100": {"ECLPATH": "/usr"}},
            "runpath_config": {"num_realizations": 200},
            "queue_config": {"max_submit": 1},
            "forward_model_steps": [{"type": "site_installed"}],
            "substitutions": {"<CONFIG_PATH>": "/project"},
            "hooked_workflows": {"PRE_SIMULATION": []},
            "active_realizations": [True, False],
            "log_path": "/project/update_log",
            "random_seed": 123456,
            "start_iteration": 0,
            "minimum_required_realizations": 1,
            "some_future_unknown_field": {"foo": "bar"},
        },
    }
    (exp_path / "index.json").write_text(json.dumps(index_data), encoding="utf-8")

    migrate(root)

    updated = json.loads((exp_path / "index.json").read_text(encoding="utf-8"))
    experiment = updated["experiment"]

    stripped_fields = {
        "some_future_unknown_field",
        "storage_path",
        "runpath_file",
        "user_config_file",
        "env_vars",
        "env_pr_fm_step",
        "runpath_config",
        "queue_config",
        "forward_model_steps",
        "substitutions",
        "hooked_workflows",
        "active_realizations",
        "log_path",
        "random_seed",
        "start_iteration",
        "minimum_required_realizations",
    }
    assert stripped_fields.isdisjoint(experiment)

    assert experiment["experiment_type"] == "Ensemble Experiment"
    assert experiment["parameter_configuration"] == [
        {"type": "gen_kw", "name": "PARAM1"}
    ]


def test_that_migration_leaves_experiment_with_only_known_fields_untouched(tmp_path):
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
            "queue_config": {"max_submit": 1},
            "random_seed": 123456,
        },
    }
    (exp_path / "index.json").write_text(json.dumps(index_data), encoding="utf-8")

    with caplog.at_level(logging.INFO):
        migrate(root)

    assert "queue_config" in caplog.text
    assert "random_seed" in caplog.text
    assert str(exp_path / "index.json") in caplog.text
