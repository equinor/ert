import pytest

from ert.config._get_update_from_options import get_update_from_options
from ert.config.parsing.config_errors import ConfigValidationError


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, "GLOBAL"),
        ({"UPDATE": "FALSE"}, None),
        ({"UPDATE": "TRUE"}, "GLOBAL"),
        ({"UPDATE": "true"}, "GLOBAL"),
        ({"UPDATE": "True"}, "GLOBAL"),
        ({"UPDATE": "false"}, None),
    ],
)
def test_that_get_update_from_options_returns_expected_strategy(options, expected):
    assert get_update_from_options(options) == expected


@pytest.mark.parametrize("value", ["INVALID", "yes", "no", "1", "0", ""])
def test_that_get_update_from_options_raises_on_invalid_value(value):
    with pytest.raises(ConfigValidationError, match=f"Invalid UPDATE option: {value}"):
        get_update_from_options({"UPDATE": value})
