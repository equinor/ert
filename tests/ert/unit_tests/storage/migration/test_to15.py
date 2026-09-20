from ert.storage.migration.to15 import add_experiment_status_to_index_json


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
