import json
from contextlib import ExitStack as does_not_raise

import pytest

from ert.run_models.everest_run_model import _get_internal_files
from everest.config import EverestConfig, WellConfig


@pytest.mark.parametrize(
    "config, expectation",
    [
        (
            {"unexpected_key": 7, "name": "well_well"},
            pytest.raises(ValueError, match="Extra inputs are not permitted"),
        ),
        (
            {"name": 7},
            pytest.raises(ValueError, match="should be a valid string"),
        ),
        (
            {"name": "well_well"},
            does_not_raise(),
        ),
        (
            {},
            pytest.raises(ValueError, match="Field required"),
        ),
        (
            {"drill_time": 10},
            pytest.raises(ValueError, match="Field required"),
        ),
        (
            {"name": "a.b"},
            pytest.raises(ValueError, match=r"Well name cannot contain any dots (.)"),
        ),
        (
            {"name": "well_well", "drill_time": -4},
            pytest.raises(ValueError, match=r"drill_time\n.*should be greater than 0"),
        ),
        (
            {"name": "well_well", "drill_time": 0},
            pytest.raises(ValueError, match=r"drill_time\n.*should be greater than 0"),
        ),
        (
            {"name": "well_well", "drill_time": "seventeen"},
            pytest.raises(ValueError, match="Input should be a valid integer"),
        ),
        (
            {"name": "well_well", "drill_time": 1.1},
            pytest.raises(ValueError, match="got a number with a fractional part"),
        ),
    ],
)
def test_well_config(config, expectation):
    with expectation:
        WellConfig(**config)


def test_that_well_names_must_be_unique(min_config):
    min_config["wells"] = [{"name": "well_well"}, {"name": "well_well"}]
    with pytest.raises(ValueError, match="Well names must be unique"):
        EverestConfig(**min_config)


@pytest.mark.parametrize(
    "config",
    [
        [{"name": "test", "drill_time": 10}],
        pytest.param([{"name": "test"}], id="Default value not in result"),
    ],
)
def test_well_config_to_wells_json(min_config, monkeypatch, tmp_path, config):
    monkeypatch.chdir(tmp_path)
    min_config["wells"] = config
    ever_config = EverestConfig(**min_config)
    for datafile, data in _get_internal_files(ever_config).items():
        datafile.parent.mkdir(exist_ok=True, parents=True)
        datafile.write_text(data, encoding="utf-8")
    with open("everest_output/.internal_data/wells.json", encoding="utf-8") as fin:
        wells_json = json.load(fin)
    assert wells_json == config


@pytest.mark.parametrize(
    "variables",
    [
        [{"name": "test", "initial_guess": 0.1}],
        [{"name": "test", "initial_guess": 0.1, "index": 1}],
        [
            {"name": "test", "initial_guess": 0.1, "index": 1},
            {"name": "test", "initial_guess": 0.1, "index": 2},
        ],
        [{"name": "test", "initial_guess": [0.1]}],
        [{"name": "test", "initial_guess": [0.1, 0.1]}],
    ],
)
def test_controls_config_to_wells_json(min_config, monkeypatch, tmp_path, variables):
    monkeypatch.chdir(tmp_path)
    min_config["controls"][0]["variables"] = variables
    ever_config = EverestConfig(**min_config)
    for datafile, data in _get_internal_files(ever_config).items():
        datafile.parent.mkdir(exist_ok=True, parents=True)
        datafile.write_text(data, encoding="utf-8")
    with open("everest_output/.internal_data/wells.json", encoding="utf-8") as fin:
        wells_json = json.load(fin)
    assert wells_json == [{"name": "test"}]
