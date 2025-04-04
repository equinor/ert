import os

import pytest
from ruamel.yaml import YAML

from ert.config.parsing.config_errors import ConfigWarning
from everest.config import EverestConfig
from tests.everest.utils import relpath


def test_input_constraint_initialization():
    cfg_dir = relpath("test_data", "mocked_test_case")
    cfg = relpath(cfg_dir, "config_input_constraints.yml")
    config = EverestConfig.load_file(cfg)
    # Check that an input constraint has been defined
    assert config.input_constraints is not None
    # Check that it is a list with two values
    assert isinstance(config.input_constraints, list)
    assert len(config.input_constraints) == 2
    # Get the first input constraint
    input_constraint = config.input_constraints[0]
    # Check that this defines both upper and lower bounds
    exp_operations = {"upper_bound", "lower_bound"}
    assert (
        exp_operations.intersection(input_constraint.model_dump(exclude_none=True))
        == exp_operations
    )
    # Check both rhs
    exp_rhs = [1, 0]
    assert [
        input_constraint.upper_bound,
        input_constraint.lower_bound,
    ] == exp_rhs
    # Check the variables
    exp_vars = ["group.w00", "group.w01", "group.w02"]
    assert set(exp_vars) == set(input_constraint.weights.keys())
    # Check the weights
    exp_weights = [0.1, 0.2, 0.3]
    assert exp_weights == [input_constraint.weights[v] for v in exp_vars]


def test_input_constraint_control_references(tmp_path, capsys, caplog):
    os.chdir(tmp_path)
    controls_config = [
        {
            "name": "dummy",
            "type": "generic_control",
            "min": 0,
            "max": 1,
            "initial_guess": 0,
            "variables": [
                {"name": "x", "index": "0"},
                {"name": "x", "index": "1"},
                {"name": "x", "index": "2"},
                {"name": "y", "index": "0"},
            ],
        }
    ]

    input_constraints_config_new = [
        {
            "weights": {
                "dummy.x.0": 1.0,
                "dummy.x.1": 1.0,
                "dummy.x.2": 1.0,
                "dummy.y.0": 1.0,
            }
        }
    ]

    input_constraints_config_deprecated = [
        {
            "weights": {
                "dummy.x-0": 1.0,
                "dummy.x-1": 1.0,
                "dummy.x-2": 1.0,
                "dummy.y-0": 1.0,
            }
        }
    ]

    yaml = YAML(typ="safe", pure=True)

    with open("config_nowarns.yml", "w+", encoding="utf-8") as f:
        yaml.dump(
            {
                "model": {"realizations": ["0", "2", "4"]},
                "controls": controls_config,
                "objective_functions": [{"name": "dummyobj"}],
                "input_constraints": input_constraints_config_new,
            },
            f,
        )

    with open("config_warns.yml", "w+", encoding="utf-8") as f:
        yaml.dump(
            {
                "model": {"realizations": ["0", "2", "4"]},
                "controls": controls_config,
                "objective_functions": [{"name": "dummyobj"}],
                "input_constraints": input_constraints_config_deprecated,
            },
            f,
        )

    # Expect no errors/warnings
    EverestConfig.load_file("config_nowarns.yml")
    out1 = capsys.readouterr().out
    assert "Deprecated input control name" not in out1

    with pytest.warns(
        ConfigWarning,
        match="Deprecated input control name: .* reference in input constraint.",
    ):
        EverestConfig.load_file("config_warns.yml")

    assert not capsys.readouterr().out
