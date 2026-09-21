import pytest

from ert.config import ConfigValidationError, LocalizationType, ParameterConfig
from ert.config.parsing.validators import validate_has_updatable_parameter


def test_that_validate_has_updatable_parameter_raises_when_no_parameters_configured():
    with pytest.raises(
        ConfigValidationError,
        match="No parameters to update as no GEN_KW, FIELD or SURFACE "
        "parameters are configured!",
    ):
        validate_has_updatable_parameter([])


def test_that_validate_has_updatable_parameter_raises_when_none_are_updatable(
    non_updatable_parameter_configs: list[ParameterConfig],
):
    with pytest.raises(
        ConfigValidationError,
        match="No parameters to update: all configured parameters have updates "
        r"disabled \(UPDATE:FALSE\)\.",
    ):
        validate_has_updatable_parameter(non_updatable_parameter_configs)


def test_that_validate_has_updatable_parameter_does_not_raise_when_one_is_updatable(
    non_updatable_parameter_configs: list[ParameterConfig],
):
    non_updatable_parameter_configs[1].update_strategy = LocalizationType.GLOBAL
    validate_has_updatable_parameter(non_updatable_parameter_configs)
