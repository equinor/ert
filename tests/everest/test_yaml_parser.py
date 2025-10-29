import os
from pathlib import Path
from textwrap import dedent

from everest.config import EverestConfig


def test_read_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("config.yml").write_text(
        dedent("""
    model: {"realizations": [0]}
    controls:
      -
        name: my_control
        type: well_control
        min: 0
        max: 0.1
        perturbation_magnitude: 0.01
        variables:
          - { name: test, initial_guess: 0.1 }
    objective_functions:
      - {name: my_objective}
    """),
        encoding="utf-8",
    )
    everest_config = EverestConfig.load_file("config.yml")
    keys = ["config_path", "controls", "model", "objective_functions"]
    assert sorted(everest_config.to_dict().keys()) == sorted(keys)

    exp_dir, exp_fn = os.path.split(os.path.realpath("config.yml"))
    assert exp_dir == everest_config.config_directory
    assert exp_fn == everest_config.config_file
