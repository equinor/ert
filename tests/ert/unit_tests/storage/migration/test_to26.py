import json

from ert.storage.migration.to26 import migrate


def test_that_migration_splits_everest_control(tmp_path):
    root = tmp_path / "project"
    root.mkdir()

    exp_path = root / "experiments" / "exp1"
    exp_path.mkdir(parents=True)

    old_everest_control = {
        "type": "everest_parameters",
        "name": "control_group",
        "forward_init": False,
        "output_file": "out.json",
        "forward_init_file": "",
        "update": False,
        "input_keys": ["k1", "k2"],
        "types": ["well_control", "generic_control"],
        "initial_guesses": [1.1, 2.2],
        "control_types": ["real", "integer"],
        "enabled": [True, False],
        "min": [0.0, 10.0],
        "max": [1.0, 20.0],
        "perturbation_types": ["absolute", "relative"],
        "perturbation_magnitudes": [0.1, 0.2],
        "scaled_ranges": [[0.1, 0.9], [11.0, 19.0]],
        "samplers": [None, {"method": "norm"}],
        "input_keys_dotdash": ["k1", "k2"],
    }

    other_param = {"type": "gen_kw", "name": "P1", "distribution": {"name": "uniform"}}
    experiment_data = {"parameter_configuration": [old_everest_control, other_param]}
    index_data = {"id": "exp-id", "experiment": experiment_data}
    (exp_path / "index.json").write_text(json.dumps(index_data), encoding="utf-8")

    migrate(root)

    updated_index = json.loads((exp_path / "index.json").read_text(encoding="utf-8"))
    new_params = updated_index["experiment"]["parameter_configuration"]
    assert len(new_params) == 3

    # Migration should only touch everest controls
    assert new_params[2] == other_param

    assert new_params[0] == {
        "type": "everest_parameters",
        "name": "k1",
        "input_key": "k1",
        "group": "control_group",
        "dimensionality": 1,
        "forward_init": False,
        "output_file": "out.json",
        "forward_init_file": "",
        "update": False,
        "control_type_": "well_control",
        "initial_guess": 1.1,
        "control_type": "real",
        "enabled": True,
        "min": 0.0,
        "max": 1.0,
        "perturbation_type": "absolute",
        "perturbation_magnitude": 0.1,
        "scaled_range": [0.1, 0.9],
        "sampler": None,
        "input_key_dotdash": "k1",
    }

    assert new_params[1] == {
        "type": "everest_parameters",
        "name": "k2",
        "input_key": "k2",
        "group": "control_group",
        "dimensionality": 1,
        "forward_init": False,
        "output_file": "out.json",
        "forward_init_file": "",
        "update": False,
        "control_type_": "generic_control",
        "initial_guess": 2.2,
        "control_type": "integer",
        "enabled": False,
        "min": 10.0,
        "max": 20.0,
        "perturbation_type": "relative",
        "perturbation_magnitude": 0.2,
        "scaled_range": [11.0, 19.0],
        "sampler": {"method": "norm"},
        "input_key_dotdash": "k2",
    }
