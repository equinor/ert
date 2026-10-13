"""Tests for validation of 'seed_strategy' in the general_input sheet."""

import pytest

from fmudesign.config_validation import validate_configuration


def _minimal_config(**extra):
    return {
        "designtype": "onebyone",
        "repeats": 10,
        "distribution_seed": 42,
        "seeds": "default",
        **extra,
    }


@pytest.mark.parametrize("value", ["bogus", ["independent"], {"joint": 1}, 5, 1.5])
def test_that_invalid_seed_strategy_raises_value_error(value):
    with pytest.raises(ValueError, match="seed_strategy"):
        validate_configuration(_minimal_config(seed_strategy=value))
