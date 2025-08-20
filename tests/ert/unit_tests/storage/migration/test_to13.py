from ert.storage.migration.to13 import migrate_gen_kw_param


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
