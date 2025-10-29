import json
import os
import uuid
from pathlib import Path

from ert.storage.migration.to15 import (
    add_experiment_status_to_index_json,
    migrate_metadata_json_into_experiment_index,
)


def test_add_experiment_status_to_index_json():
    original_index = {
        "id": "my_experiment_id",
        "name": "my_experiment_name",
        "ensembles": [
            "ensemble_1",
            "ensemble_2",
        ],
    }
    expected_index = {
        "id": "my_experiment_id",
        "name": "my_experiment_name",
        "ensembles": [
            "ensemble_1",
            "ensemble_2",
        ],
        "status": {"message": "", "status": "completed"},
    }

    migrated_index = add_experiment_status_to_index_json(original_index)
    assert migrated_index == expected_index


def test_that_metadata_json_is_written_to_experiment_index_after_migration(
    use_tmpdir,
):
    with open("index.json", "w", encoding="utf-8") as f:
        json.dump({"version": 13, "migrations": []}, f, indent=2)

    os.mkdir("experiments")
    os.mkdir("ensembles")

    exp_id = uuid.uuid1(0)
    exp_path = Path("experiments") / str(exp_id)
    os.mkdir(exp_path)
    with open(Path(exp_path, "index.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "id": str(exp_id),
                "name": "exp",
            },
            f,
            indent=2,
        )

    with open(exp_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {"weights": "4,2,1"},
            f,
            indent=2,
        )

    migrate_metadata_json_into_experiment_index(Path.cwd())

    assert not (exp_path / "metadata.json").exists()
    with open(exp_path / "index.json", encoding="utf-8") as f:
        experiment_json = json.load(f)
        assert experiment_json["experiment"]["weights"] == "4,2,1"


def test_that_empty_metadata_is_still_deleted_after_migration(
    use_tmpdir,
):
    with open("index.json", "w", encoding="utf-8") as f:
        json.dump({"version": 13, "migrations": []}, f, indent=2)

    os.mkdir("experiments")
    os.mkdir("ensembles")

    exp_id = uuid.uuid1(0)
    exp_path = Path("experiments") / str(exp_id)
    os.mkdir(exp_path)
    with open(Path(exp_path, "index.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "id": str(exp_id),
                "name": "exp",
            },
            f,
            indent=2,
        )

    with open(exp_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {},
            f,
            indent=2,
        )

    migrate_metadata_json_into_experiment_index(Path.cwd())

    assert not (exp_path / "metadata.json").exists()
    with open(exp_path / "index.json", encoding="utf-8") as f:
        experiment_json = json.load(f)
        assert experiment_json["experiment"] == {}
