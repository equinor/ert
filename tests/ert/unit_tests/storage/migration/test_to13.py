import json
import os
import uuid
from pathlib import Path

from ert.storage.migration.to13 import (
    migrate_gen_kw_param,
    point_experiments_to_ensembles,
)


def test_that_migrate_genkw_parameters_maps_tfds_to_single_param_instances():
    original_gen_kw = {
        "COEFFS": {
            "name": "COEFFS",
            "forward_init": False,
            "update": False,
            "transform_function_definitions": [
                {"name": "a", "param_name": "UNIFORM", "values": ["0", "1"]},
                {"name": "b", "param_name": "RAW", "values": []},
                {"name": "c", "param_name": "LOGNORMAL", "values": ["0", "2"]},
            ],
            "type": "gen_kw",
        }
    }

    migrated = migrate_gen_kw_param(original_gen_kw)

    assert set(migrated.keys()) == {"a", "b", "c"}
    assert migrated["a"] == {
        "name": "a",
        "type": "gen_kw",
        "group": "COEFFS",
        "distribution": {"name": "uniform", "min": "0", "max": "1"},
        "forward_init": False,
        "update": False,
        "input_source": "sampled",
    }
    assert migrated["b"] == {
        "name": "b",
        "type": "gen_kw",
        "group": "COEFFS",
        "distribution": {"name": "raw"},
        "forward_init": False,
        "update": False,
        "input_source": "design_matrix",
    }
    assert migrated["c"] == {
        "name": "c",
        "type": "gen_kw",
        "group": "COEFFS",
        "distribution": {"name": "lognormal", "mean": "0", "std": "2"},
        "forward_init": False,
        "update": False,
        "input_source": "sampled",
    }


def setup_experiments_and_ensembles(
    experiment_uuids: list[uuid.UUID],
    ensemble_uuids: list[uuid.UUID],
    ensemble_mapping: dict[uuid.UUID, uuid.UUID] | None = None,
) -> None:
    with open("index.json", "w", encoding="utf-8") as f:
        json.dump({"version": 12, "migrations": []}, f, indent=2)

    os.mkdir("experiments")
    os.mkdir("ensembles")

    for i, exp_id in enumerate(experiment_uuids):
        exp_path = Path("experiments", str(exp_id))
        os.mkdir(exp_path)
        with open(Path(exp_path, "index.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "id": str(exp_id),
                    "name": f"exp_{i}",
                },
                f,
                indent=2,
            )

    for ens_id in ensemble_uuids:
        if ensemble_mapping is None:
            exp_id = experiment_uuids[ensemble_uuids.index(ens_id)]
        else:
            exp_id = ensemble_mapping.get(ens_id)
            if exp_id is None:
                continue

        ens_path = Path("ensembles", str(ens_id))
        os.mkdir(ens_path)
        with open(Path(ens_path) / "index.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "id": str(ens_id),
                    "experiment_id": str(exp_id),
                    "ensemble_size": 100,
                    "iteration": 0,
                    "name": "default",
                    "prior_ensemble_id": None,
                    "started_at": "2025-09-22T10:16:23.787333",
                    "everest_realization_info": None,
                },
                f,
                indent=2,
            )


def test_that_experiments_point_to_ensembles_after_migration_one_to_one(use_tmpdir):
    ensemble_uuids = [uuid.uuid1(i) for i in range(10)]
    experiment_uuids = [uuid.uuid1(i + 10) for i in range(10)]

    setup_experiments_and_ensembles(experiment_uuids, ensemble_uuids)
    point_experiments_to_ensembles(Path.cwd())

    for ens_id, exp_id in zip(ensemble_uuids, experiment_uuids, strict=False):
        with (
            open(
                Path("ensembles") / str(ens_id) / "index.json", encoding="utf-8"
            ) as ens_f,
            open(
                Path("experiments") / str(exp_id) / "index.json", encoding="utf-8"
            ) as exp_f,
        ):
            ens_data = json.load(ens_f)
            exp_data = json.load(exp_f)
            assert exp_data["ensembles"] == [str(ens_id)]
            assert ens_data["experiment_id"] == str(exp_id)


def test_that_experiments_point_to_ensembles_after_migration_many_ensembles_per_exp(
    use_tmpdir,
):
    num_ensembles = 500
    num_experiments = 50
    ensemble_uuids = [uuid.uuid1(i) for i in range(num_ensembles)]
    experiment_uuids = [uuid.uuid1(num_ensembles + i) for i in range(num_experiments)]

    ensemble_mapping = {
        ens_id: experiment_uuids[i % num_experiments]
        for i, ens_id in enumerate(ensemble_uuids)
    }
    expected_exp_to_ens = {str(exp_id): [] for exp_id in experiment_uuids}
    for ens_id, exp_id in ensemble_mapping.items():
        expected_exp_to_ens[str(exp_id)].append(str(ens_id))

    setup_experiments_and_ensembles(experiment_uuids, ensemble_uuids, ensemble_mapping)
    point_experiments_to_ensembles(Path.cwd())

    for exp_id in experiment_uuids:
        with open(
            Path("experiments") / str(exp_id) / "index.json", encoding="utf-8"
        ) as exp_f:
            exp_data = json.load(exp_f)
            assert exp_data["ensembles"] == sorted(expected_exp_to_ens[str(exp_id)])
            for ensemble_id in exp_data["ensembles"]:
                with open(
                    Path("ensembles") / ensemble_id / "index.json", encoding="utf-8"
                ) as ens_check_f:
                    ens_check_data = json.load(ens_check_f)
                    assert ens_check_data["experiment_id"] == str(exp_id)


def test_that_experiments_point_to_ensembles_after_migration_for_empty_experiment(
    use_tmpdir,
):
    num_ensembles = 500
    num_experiments = 50
    ensemble_uuids = [uuid.uuid1(i) for i in range(num_ensembles)]
    experiment_uuids = [uuid.uuid1(num_ensembles + i) for i in range(num_experiments)]
    empty_experiment_uuid = experiment_uuids[0]

    ensemble_mapping = {
        ens_id: experiment_uuids[i % num_experiments]
        for i, ens_id in enumerate(ensemble_uuids)
        if experiment_uuids[i % num_experiments] != empty_experiment_uuid
    }

    setup_experiments_and_ensembles(experiment_uuids, ensemble_uuids, ensemble_mapping)
    point_experiments_to_ensembles(Path.cwd())

    with open(
        Path("experiments") / str(empty_experiment_uuid) / "index.json",
        encoding="utf-8",
    ) as exp_f:
        exp_data = json.load(exp_f)
        assert exp_data["ensembles"] == []
