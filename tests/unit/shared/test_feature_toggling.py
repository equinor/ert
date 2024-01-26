import os
from unittest.mock import Mock

import pytest

import ert.__main__
from ert.__main__ import ert_parser
from ert.cli import TEST_RUN_MODE
from ert.shared.feature_toggling import FeatureToggling


def is_default(feature_name: str) -> bool:
    return (
        FeatureToggling._conf[feature_name].value
        == FeatureToggling._conf_original[feature_name].value
    )


def feature_to_env_name(feature_name: str) -> str:
    return f"ERT_FEATURE_{feature_name.replace('-', '_').upper()}"


@pytest.fixture(autouse=True)
def mocked_valid_file(monkeypatch):
    monkeypatch.setattr(
        ert.__main__, "valid_file", Mock(return_value="not_a_real_config.ert")
    )


@pytest.fixture(autouse=True)
def reset_feature_toggling():
    FeatureToggling.reset()


@pytest.fixture(autouse=True)
def reset_environment_vars(monkeypatch):
    monkeypatch.setattr(os, "environ", {})


def test_feature_toggling_from_env():
    os.environ["ERT_FEATURE_SCHEDULER"] = "True"

    parsed = ert_parser(
        None,
        [
            TEST_RUN_MODE,
            "not_a_real_config.ert",
        ],
    )
    FeatureToggling.update_from_args(parsed)

    assert FeatureToggling.is_enabled("scheduler")


def test_feature_toggling_from_args():
    parsed = ert_parser(
        None,
        [
            TEST_RUN_MODE,
            "--disable-scheduler",
            "not_a_real_config.ert",
        ],
    )
    FeatureToggling.update_from_args(parsed)

    assert not FeatureToggling.is_enabled("scheduler")


@pytest.mark.parametrize(
    "environment_vars, arguments, expected",
    [
        (
            {"ERT_FEATURE_SCHEDULER": "False"},
            ["--enable-scheduler"],
            {"scheduler": True},
        ),
        (
            {"ERT_FEATURE_SCHEDULER": "True"},
            ["--disable-scheduler"],
            {"scheduler": False},
        ),
        (
            {"ERT_FEATURE_SCHEDULER": ""},
            ["--disable-scheduler"],
            {"scheduler": False},
        ),
    ],
)
def test_feature_toggling_both(environment_vars, arguments, expected):
    for key, value in environment_vars.items():
        os.environ[key] = value

    parsed = ert_parser(
        None,
        [
            TEST_RUN_MODE,
            *arguments,
            "not_a_real_config.ert",
        ],
    )

    FeatureToggling.update_from_args(parsed)

    for key, value in expected.items():
        assert FeatureToggling.value(key) == value


@pytest.mark.parametrize(
    "feature_name, value",
    [
        ("scheduler", "incorrect"),
    ],
)
def test_feature_toggling_incorrect_input(feature_name, value):
    os.environ[feature_to_env_name(feature_name)] = value
    parsed = ert_parser(
        None,
        [
            TEST_RUN_MODE,
            "not_a_real_config.ert",
        ],
    )

    FeatureToggling.update_from_args(parsed)
    assert is_default(feature_name)
