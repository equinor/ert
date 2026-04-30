import pytest

from ert.config._get_update_from_options import get_update_from_options
from ert.config.parsing import ConfigValidationError


@pytest.mark.parametrize(
    ("options", "default", "expected"),
    [
        ({}, None, None),
        ({}, "ADAPTIVE", "ADAPTIVE"),
        ({"UPDATE": "ADAPTIVE"}, None, "ADAPTIVE"),
        ({"UPDATE": "DISTANCE"}, None, "DISTANCE"),
        ({"UPDATE": "adaptive"}, None, "ADAPTIVE"),
        ({"UPDATE": "distance"}, None, "DISTANCE"),
        ({"UPDATE": "TRUE"}, None, "ADAPTIVE"),
        ({"UPDATE": "true"}, None, "ADAPTIVE"),
        ({"UPDATE": "True"}, None, "ADAPTIVE"),
        ({"UPDATE": "FALSE"}, None, None),
        ({"UPDATE": "false"}, None, None),
        ({"UPDATE": "NONE"}, None, None),
        ({"UPDATE": "none"}, None, None),
        ({"UPDATE": "None"}, None, None),
        ({"UPDATE": "NONE"}, "ADAPTIVE", None),
        ({"UPDATE": "FALSE"}, "ADAPTIVE", None),
    ],
)
def test_that_get_update_from_options_returns_expected_value(
    options, default, expected
):
    assert get_update_from_options(options, default) == expected


def test_that_get_update_from_options_raises_on_unknown_strategy():
    with pytest.raises(ConfigValidationError, match="Unknown UPDATE value"):
        get_update_from_options({"UPDATE": "INVALID_STRATEGY"})
