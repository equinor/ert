import pytest

from ert.config import GenKwConfig, LocalizationType
from ert.gui.ertwidgets.models.parameter_configuration_state_model import (
    ParameterConfigurationStateModel,
)


def _parameter(
    name: str, update_strategy: LocalizationType | None = LocalizationType.GLOBAL
) -> GenKwConfig:
    return GenKwConfig(
        name=name,
        distribution={"name": "uniform", "min": 0, "max": 1},
        update_strategy=update_strategy,
    )


@pytest.mark.parametrize("with_prior", [False, True])
def test_that_applying_update_strategies_does_not_mutate_source_parameters(
    with_prior: bool,
):
    source = [_parameter("source")]
    state = ParameterConfigurationStateModel([_parameter("configured")])
    if with_prior:
        state.select_prior(source)
    else:
        state = ParameterConfigurationStateModel(source)

    state.apply_update_strategies({"GEN_KW": LocalizationType.ADAPTIVE})

    assert source[0].update_strategy == LocalizationType.GLOBAL
    assert state.parameters[0].update_strategy == LocalizationType.ADAPTIVE


def test_that_applying_update_strategies_with_prior_leaves_configured_draft_unset():
    state = ParameterConfigurationStateModel([_parameter("configured")])
    state.select_prior([_parameter("prior")])

    state.apply_update_strategies({"GEN_KW": LocalizationType.ADAPTIVE})

    assert state._configured_parameters.draft is None
    assert state._prior_parameters is not None
    assert state._prior_parameters.draft is not None


def test_that_parameters_without_update_strategy_keep_none_strategy():
    state = ParameterConfigurationStateModel(
        [_parameter("without", update_strategy=None), _parameter("with")]
    )

    state.apply_update_strategies({"GEN_KW": LocalizationType.ADAPTIVE})

    assert [p.update_strategy for p in state.parameters] == [
        None,
        LocalizationType.ADAPTIVE,
    ]


def test_that_update_strategies_only_include_parameters_with_a_strategy():
    state = ParameterConfigurationStateModel(
        [_parameter("without", update_strategy=None), _parameter("with")]
    )

    assert state.update_strategies == {"GEN_KW": LocalizationType.GLOBAL}


def test_that_update_strategies_reflect_applied_draft():
    state = ParameterConfigurationStateModel([_parameter("configured")])

    state.apply_update_strategies({"GEN_KW": LocalizationType.ADAPTIVE})

    assert state.update_strategies == {"GEN_KW": LocalizationType.ADAPTIVE}


def test_that_selecting_and_deselecting_prior_switches_active_parameters():
    configured = [_parameter("configured")]
    prior = [_parameter("prior")]
    state = ParameterConfigurationStateModel(configured)

    state.select_prior(prior)
    assert state.parameters == prior

    state.deselect_prior()
    assert state.parameters == configured
    assert state._prior_parameters is None


def test_that_reset_clears_configured_and_prior_drafts():
    configured = [_parameter("configured")]
    prior = [_parameter("prior")]
    state = ParameterConfigurationStateModel(configured)
    state.apply_update_strategies({"GEN_KW": LocalizationType.ADAPTIVE})
    state.select_prior(prior)
    state.apply_update_strategies({"GEN_KW": LocalizationType.ADAPTIVE})

    state.reset_parameters()

    assert state._configured_parameters.draft is None
    assert state._prior_parameters is not None
    assert state._prior_parameters.draft is None
    assert state.parameters == prior


def test_that_reselecting_an_equal_prior_keeps_its_draft():
    state = ParameterConfigurationStateModel([_parameter("configured")])
    state.select_prior([_parameter("prior")])
    state.apply_update_strategies({"GEN_KW": LocalizationType.ADAPTIVE})

    state.select_prior([_parameter("prior")])

    assert state.update_strategies == {"GEN_KW": LocalizationType.ADAPTIVE}


def test_that_selecting_a_different_prior_discards_previous_prior_draft():
    state = ParameterConfigurationStateModel([_parameter("configured")])
    state.select_prior([_parameter("prior")])
    state.apply_update_strategies({"GEN_KW": LocalizationType.ADAPTIVE})

    state.select_prior([_parameter("other_prior")])

    assert state.update_strategies == {"GEN_KW": LocalizationType.GLOBAL}
