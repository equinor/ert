import pytest

from ert.config import ConfigValidationError, GenKwConfig, LocalizationType
from ert.config.parsing.validators import validate_has_updatable_parameter


def _gen_kw_config(name: str, update_strategy: LocalizationType | None) -> GenKwConfig:
    return GenKwConfig(
        name=name,
        distribution={"name": "normal", "mean": 0, "std": 1},
        update_strategy=update_strategy,
    )


def test_that_validate_has_updatable_parameter_raises_when_no_parameters_configured():
    with pytest.raises(
        ConfigValidationError,
        match="No parameters to update as no GEN_KW, FIELD or SURFACE "
        "parameters are configured!",
    ):
        validate_has_updatable_parameter([])


@pytest.mark.parametrize(
    "parameter_configs",
    [
        [_gen_kw_config("COEFFS", None)],
        [_gen_kw_config("COEFFS_A", None), _gen_kw_config("COEFFS_B", None)],
    ],
)
def test_that_validate_has_updatable_parameter_raises_when_none_are_updatable(
    parameter_configs,
):
    with pytest.raises(
        ConfigValidationError,
        match="No parameters to update as all parameters were set to update:false!",
    ):
        validate_has_updatable_parameter(parameter_configs)


def test_that_validate_has_updatable_parameter_does_not_raise_when_one_is_updatable():
    parameter_configs = [
        _gen_kw_config("COEFFS_A", None),
        _gen_kw_config("COEFFS_B", LocalizationType.GLOBAL),
    ]
    validate_has_updatable_parameter(parameter_configs)
