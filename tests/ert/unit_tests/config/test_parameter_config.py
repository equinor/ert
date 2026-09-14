import pytest

from ert.config import LocalizationType, ParameterConfig, has_updatable_parameters


def test_that_has_updatable_parameters_is_false_for_empty_parameter_list():
    assert not has_updatable_parameters([])


@pytest.mark.parametrize(
    "parameter_indices",
    [[0], [1], [2], [0, 1, 2]],
    ids=["GEN_KW", "FIELD", "SURFACE", "mixed"],
)
def test_that_has_updatable_parameters_is_false_when_all_parameters_disable_update(
    non_updatable_parameter_configs: list[ParameterConfig],
    parameter_indices: list[int],
):
    parameter_configs = [
        non_updatable_parameter_configs[index] for index in parameter_indices
    ]
    assert not has_updatable_parameters(parameter_configs)


@pytest.mark.parametrize("update_strategy", list(LocalizationType))
@pytest.mark.parametrize(
    "updatable_index", [0, 1, 2], ids=["GEN_KW", "FIELD", "SURFACE"]
)
@pytest.mark.parametrize(
    "include_non_updatable", [False, True], ids=["single", "mixed"]
)
def test_that_has_updatable_parameters_is_true_when_any_parameter_is_updatable(
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
    assert has_updatable_parameters(parameter_configs)
