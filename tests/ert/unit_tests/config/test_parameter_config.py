import pytest

from ert.config import GenKwConfig, LocalizationType, has_updatable_parameters


def _gen_kw_config(name: str, update_strategy: LocalizationType | None) -> GenKwConfig:
    return GenKwConfig(
        name=name,
        distribution={"name": "normal", "mean": 0, "std": 1},
        update_strategy=update_strategy,
    )


def test_that_has_updatable_parameters_is_false_for_empty_parameter_list():
    assert not has_updatable_parameters([])


def test_that_has_updatable_parameters_is_false_when_all_parameters_disable_update():
    parameter_configs = [
        _gen_kw_config("COEFFS_A", None),
        _gen_kw_config("COEFFS_B", None),
    ]
    assert not has_updatable_parameters(parameter_configs)


@pytest.mark.parametrize(
    "update_strategy",
    [
        LocalizationType.GLOBAL,
        LocalizationType.ADAPTIVE,
        LocalizationType.DISTANCE,
    ],
)
def test_that_has_updatable_parameters_is_true_when_any_parameter_is_updatable(
    update_strategy,
):
    parameter_configs = [
        _gen_kw_config("COEFFS_A", None),
        _gen_kw_config("COEFFS_B", update_strategy),
    ]
    assert has_updatable_parameters(parameter_configs)
