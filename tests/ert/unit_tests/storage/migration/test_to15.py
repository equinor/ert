from ert.storage.migration.to15 import update_json


def test_update_json():
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

    migrated_index = update_json(original_index)
    assert migrated_index == expected_index
