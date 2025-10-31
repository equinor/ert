import pytest
from ruamel.yaml import YAML

from ert.config import ConfigWarning
from everest.config import EverestConfig, InputConstraintConfig, OptimizationConfig
from tests.everest.conftest import everest_config_with_defaults


def test_input_constraint_control_references(tmp_path, capsys, caplog, monkeypatch):
    monkeypatch.chdir(tmp_path)
    controls_config = [
        {
            "name": "dummy",
            "type": "generic_control",
            "min": 0,
            "max": 1,
            "initial_guess": 0,
            "perturbation_magnitude": 0.01,
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


def test_that_auto_scale_and_input_constraints_scale_are_mutually_exclusive(tmp_path):
    with pytest.raises(
        ValueError,
        match=(
            "The auto_scale option in the optimization section and the scale "
            "options in the input_constraints section are mutually exclusive"
        ),
    ):
        everest_config_with_defaults(
            optimization=OptimizationConfig(auto_scale=True),
            input_constraints=[
                InputConstraintConfig.model_validate(
                    {"upper_bound": 1.0, "weights": {"a": 1.0, "b": 1.0}, "scale": 2.0}
                )
                for i in range(2)
            ],
        )
