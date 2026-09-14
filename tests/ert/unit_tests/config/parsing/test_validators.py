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


@pytest.mark.parametrize(
    "parameter_indices",
    [[0], [1], [2], [0, 1, 2]],
    ids=["GEN_KW", "FIELD", "SURFACE", "mixed"],
)
def test_that_validate_has_updatable_parameter_raises_when_none_are_updatable(
    non_updatable_parameter_configs: list[ParameterConfig],
    parameter_indices: list[int],
):
    parameter_configs = [
        non_updatable_parameter_configs[index] for index in parameter_indices
    ]
    with pytest.raises(
        ConfigValidationError,
        match="No parameters to update as all parameters were set to update:false!",
    ):
        validate_has_updatable_parameter(parameter_configs)


@pytest.mark.parametrize("update_strategy", list(LocalizationType))
@pytest.mark.parametrize(
    "updatable_index", [0, 1, 2], ids=["GEN_KW", "FIELD", "SURFACE"]
)
@pytest.mark.parametrize(
    "include_non_updatable", [False, True], ids=["single", "mixed"]
)
def test_that_validate_has_updatable_parameter_does_not_raise_when_one_is_updatable(
    non_updatable_parameter_configs: list[ParameterConfig],
    updatable_index: int,
    update_strategy: LocalizationType,
    include_non_updatable: bool,
):
    updatable_parameter = non_updatable_parameter_configs[updatable_index]
    updatable_parameter.update_strategy = update_strategy
    parameter_configs = (
        non_updatable_parameter_configs
        if include_non_updatable
        else [updatable_parameter]
    )
    validate_has_updatable_parameter(parameter_configs)
