import pytest
from ruamel.yaml import YAML

from ert.config.parsing.config_errors import ConfigWarning
from everest.config import EverestConfig


def test_input_constraint_control_references(tmp_path, capsys, caplog, monkeypatch):
    monkeypatch.chdir(tmp_path)
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
