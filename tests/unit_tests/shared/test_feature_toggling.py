import os
from unittest.mock import Mock

import pytest

import ert.__main__
from ert.__main__ import ert_parser
from ert.cli import TEST_RUN_MODE
from ert.shared.feature_toggling import FeatureScheduler


@pytest.fixture(autouse=True)
def mocked_valid_file(monkeypatch):
    monkeypatch.setattr(
        ert.__main__, "valid_file", Mock(return_value="not_a_real_config.ert")
    )


@pytest.fixture(autouse=True)
def reset_feature_toggling(monkeypatch):
    monkeypatch.setattr(FeatureScheduler, "_value", None)


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
    FeatureScheduler.set_value(parsed)
    assert FeatureScheduler._value is True


def test_feature_toggling_from_args():
    parsed = ert_parser(
        None,
        [
            TEST_RUN_MODE,
            "--disable-scheduler",
            "not_a_real_config.ert",
        ],
    )
    FeatureScheduler.set_value(parsed)
    assert FeatureScheduler._value is False


@pytest.mark.parametrize(
    "environment_vars, arguments, expected",
    [
        (
            {"ERT_FEATURE_SCHEDULER": "False"},
            ["--enable-scheduler"],
            True,
        ),
        (
            {"ERT_FEATURE_SCHEDULER": "True"},
            ["--disable-scheduler"],
            False,
        ),
        (
            {"ERT_FEATURE_SCHEDULER": ""},
            ["--disable-scheduler"],
            False,
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
    FeatureScheduler.set_value(parsed)
    assert FeatureScheduler._value is expected


def test_feature_toggling_incorrect_input(monkeypatch):
    monkeypatch.setenv("ERT_FEATURE_SCHEDULER", "incorrect")
    parsed = ert_parser(
        None,
        [
            TEST_RUN_MODE,
            "not_a_real_config.ert",
        ],
    )
    with pytest.raises(ValueError):
        FeatureScheduler.set_value(parsed)
    assert FeatureScheduler._value is None
