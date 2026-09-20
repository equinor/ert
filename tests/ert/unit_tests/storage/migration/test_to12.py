from ert.storage.migration.to12 import migrate_everest_param


def test_migrate_everest_param_listkeys():
    original = {"name": "supername", "input_keys": ["supername.2", "supername.3"]}
    migrated = migrate_everest_param(original)

    assert migrated["input_keys"] == original["input_keys"]


def test_migrate_everest_param_dict_of_listkeys():
    original = {
        "name": "supername",
        "input_keys": {
            "WELL-1": ["1", "2", "3", "4"],
            "WELL-2": ["1", "2", "3", "4"],
            "WELL-3": ["1", "2", "3", "4"],
        },
    }

    migrated = migrate_everest_param(original)

    assert migrated["input_keys"] == [
        "supername.WELL-1.1",
        "supername.WELL-1.2",
        "supername.WELL-1.3",
        "supername.WELL-1.4",
        "supername.WELL-2.1",
        "supername.WELL-2.2",
        "supername.WELL-2.3",
        "supername.WELL-2.4",
        "supername.WELL-3.1",
        "supername.WELL-3.2",
        "supername.WELL-3.3",
        "supername.WELL-3.4",
    ]
