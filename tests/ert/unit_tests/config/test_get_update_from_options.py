import pytest

from ert.config._get_update_from_options import get_update_from_options
from ert.config.parsing import ConfigValidationError


@pytest.mark.parametrize(
    ("options", "default", "expected"),
    [
        ({}, None, None),
        ({}, "GLOBAL", "GLOBAL"),
        ({}, "DISTANCE", "DISTANCE"),
        ({}, "ADAPTIVE", "ADAPTIVE"),
        ({}, "TRUE", "GLOBAL"),
        ({}, "true", "GLOBAL"),
        ({}, "FALSE", None),
        ({}, "false", None),
        ({"UPDATE": "FALSE"}, "GLOBAL", None),
        ({"UPDATE": "None"}, "GLOBAL", None),
        ({"UPDATE": "DISTANCE"}, "GLOBAL", "DISTANCE"),
        ({"UPDATE": "GLOBAL"}, None, "GLOBAL"),
        ({"UPDATE": "DISTANCE"}, None, "DISTANCE"),
        ({"UPDATE": "ADAPTIVE"}, None, "ADAPTIVE"),
        ({"UPDATE": "global"}, None, "GLOBAL"),
        ({"UPDATE": "distance"}, None, "DISTANCE"),
        ({"UPDATE": "adaptive"}, None, "ADAPTIVE"),
        ({"UPDATE": "TRUE"}, None, "GLOBAL"),
        ({"UPDATE": "true"}, None, "GLOBAL"),
        ({"UPDATE": "True"}, None, "GLOBAL"),
        ({"UPDATE": "FALSE"}, None, None),
        ({"UPDATE": "false"}, None, None),
        ({"UPDATE": "NONE"}, None, None),
        ({"UPDATE": "none"}, None, None),
        ({"UPDATE": "None"}, None, None),
    ],
)
def test_that_get_update_from_options_returns_expected_strategy(
    options, default, expected
):
    assert get_update_from_options(options, default) == expected


def test_that_get_update_from_options_raises_on_unknown_value():
    with pytest.raises(
        ConfigValidationError,
        match="Unknown UPDATE value: INVALID",
    ):
        get_update_from_options({"UPDATE": "INVALID"})


def test_that_get_update_from_options_raises_on_unknown_default():
    with pytest.raises(
        ConfigValidationError,
        match="Unknown UPDATE value: BOGUS",
    ):
        get_update_from_options({}, "BOGUS")
