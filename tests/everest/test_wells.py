from contextlib import ExitStack as does_not_raise

import pytest

from everest.config import EverestConfig, WellConfig
from everest.config_file_loader import yaml_file_to_substituted_config_dict
from tests.everest.utils import relpath


@pytest.fixture
def mocked_config():
    return yaml_file_to_substituted_config_dict(
        relpath("test_data", "mocked_test_case", "mocked_test_case.yml")
    )


def test_lint_mocked_config(mocked_config):
    EverestConfig(**mocked_config)


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
            pytest.raises(ValueError, match=r"Well name can not contain any dots (.)"),
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
