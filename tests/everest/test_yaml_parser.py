import os
from pathlib import Path
from textwrap import dedent

import pytest

from ert.config.parsing import ConfigKeys as ErtConfigKeys
from everest.config import EverestConfig
from everest.simulator.everest_to_ert import (
    everest_to_ert_config_dict,
)


@pytest.mark.parametrize("random_seed", [None, 1234])
def test_random_seed(tmp_path, monkeypatch, random_seed):
    monkeypatch.chdir(tmp_path)
    config = {"model": {"realizations": [0]}}
    if random_seed:
        config["environment"] = {"random_seed": random_seed}
    ever_config = EverestConfig.with_defaults(**config)
    dictionary = everest_to_ert_config_dict(ever_config)

    if random_seed is None:
        assert ever_config.environment.random_seed > 0
        assert dictionary[ErtConfigKeys.RANDOM_SEED] > 0
    else:
        assert ever_config.environment.random_seed == random_seed
        assert dictionary[ErtConfigKeys.RANDOM_SEED] == random_seed


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
