"""Tests for validation of 'seed_strategy' in the general_input sheet."""

import pytest

from fmudesign.config_validation import SeedStrategy, validate_configuration


def _minimal_config(**extra):
    return {
        "designtype": "onebyone",
        "repeats": 10,
        "distribution_seed": 42,
        "seeds": "default",
        **extra,
    }


def test_that_seed_strategy_defaults_to_joint():
    cfg = validate_configuration(_minimal_config())
    assert cfg["seed_strategy"] is SeedStrategy.JOINT


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("independent", SeedStrategy.INDEPENDENT),
        ("Independent", SeedStrategy.INDEPENDENT),
        ("INDEPENDENT", SeedStrategy.INDEPENDENT),
        (" independent ", SeedStrategy.INDEPENDENT),
        (None, SeedStrategy.JOINT),
        ("None", SeedStrategy.JOINT),
    ],
)
def test_that_seed_strategy_input_is_normalized_to_enum(value, expected):
    cfg = validate_configuration(_minimal_config(seed_strategy=value))
    assert cfg["seed_strategy"] is expected


@pytest.mark.parametrize("value", ["bogus", ["independent"], {"joint": 1}, 5, 1.5])
def test_that_invalid_seed_strategy_raises_value_error(value):
    with pytest.raises(ValueError, match="seed_strategy"):
        validate_configuration(_minimal_config(seed_strategy=value))
