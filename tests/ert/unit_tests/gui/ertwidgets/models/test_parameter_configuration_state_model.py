import pytest
from pytestqt.qtbot import QtBot

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


def test_that_applying_update_strategies_does_not_mutate_source_parameters():
    source = [_parameter("source")]
    state = ParameterConfigurationStateModel(source)

    state.apply_update_strategies({"GEN_KW": LocalizationType.ADAPTIVE})

    assert source[0].update_strategy == LocalizationType.GLOBAL
    assert state.parameters[0].update_strategy == LocalizationType.ADAPTIVE


@pytest.mark.parametrize(
    "prior",
    [
        pytest.param([_parameter("prior")], id="prior"),
        pytest.param([], id="empty_prior"),
    ],
)
def test_that_updating_prior_parameter_config_raises_error_and_keeps_prior(
    qtbot: QtBot, prior: list[GenKwConfig]
):
    state = ParameterConfigurationStateModel([_parameter("configured")])
    state.select_prior(prior)

    with (
        qtbot.assertNotEmitted(state.changed),
        pytest.raises(RuntimeError, match="cannot be changed"),
    ):
        state.apply_update_strategies({"GEN_KW": LocalizationType.ADAPTIVE})

    assert state.parameters == prior


def test_that_selecting_an_empty_prior_gives_no_parameters():
    state = ParameterConfigurationStateModel([_parameter("configured")])

    state.select_prior([])

    assert state.parameters == []
    assert state.update_strategies == {}


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
    assert state.has_prior

    state.deselect_prior()
    assert state.parameters == configured
    assert not state.has_prior
    assert state._prior_parameters is None


def test_that_reset_restores_configured_update_strategies():
    configured = [_parameter("configured")]
    state = ParameterConfigurationStateModel(configured)
    state.apply_update_strategies({"GEN_KW": LocalizationType.ADAPTIVE})

    state.reset_parameters()

    assert state.parameters == configured
    assert state.update_strategies == {"GEN_KW": LocalizationType.GLOBAL}


def test_that_deselecting_prior_restores_configured_draft():
    state = ParameterConfigurationStateModel([_parameter("configured")])
    state.apply_update_strategies({"GEN_KW": LocalizationType.ADAPTIVE})

    state.select_prior([_parameter("prior")])
    state.deselect_prior()

    assert state.update_strategies == {"GEN_KW": LocalizationType.ADAPTIVE}
