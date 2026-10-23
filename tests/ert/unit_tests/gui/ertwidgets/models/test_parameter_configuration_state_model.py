import pytest
from pytestqt.qtbot import QtBot

from ert.config import GenKwConfig, LocalizationType, ParameterType, SurfaceConfig
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


def _surface(
    name: str, update_strategy: LocalizationType | None = LocalizationType.GLOBAL
) -> SurfaceConfig:
    return SurfaceConfig(
        name=name,
        forward_init=False,
        update_strategy=update_strategy,
        ncol=1,
        nrow=1,
        xori=0,
        yori=0,
        xinc=1,
        yinc=1,
        rotation=0,
        yflip=1,
        forward_init_file="init_%d.irap",
        output_file="output.irap",
        base_surface_path="base.irap",
    )


def test_that_applying_update_strategies_emits_changed(qtbot: QtBot):
    state = ParameterConfigurationStateModel([_parameter("configured")])

    with qtbot.waitSignal(state.changed, timeout=0):
        state.apply_update_strategies({ParameterType.GEN_KW: LocalizationType.ADAPTIVE})


def test_that_parameters_with_type_missing_from_mapping_keep_their_update_strategy():
    state = ParameterConfigurationStateModel(
        [_parameter("gen_kw"), _surface("surface", LocalizationType.DISTANCE)]
    )

    state.apply_update_strategies({ParameterType.GEN_KW: LocalizationType.ADAPTIVE})

    assert [p.update_strategy for p in state.parameters] == [
        LocalizationType.ADAPTIVE,
        LocalizationType.DISTANCE,
    ]


def test_that_applying_update_strategies_does_not_mutate_source_parameters():
    source = [_parameter("source")]
    state = ParameterConfigurationStateModel(source)

    state.apply_update_strategies({ParameterType.GEN_KW: LocalizationType.ADAPTIVE})

    assert source[0].update_strategy == LocalizationType.GLOBAL
    assert state.parameters[0].update_strategy == LocalizationType.ADAPTIVE


@pytest.mark.parametrize(
    "prior",
    [
        pytest.param([_parameter("prior")], id="prior"),
        pytest.param([], id="empty_prior"),
    ],
)
def test_that_applying_update_strategies_with_prior_selected_raises_and_emits_nothing(
    qtbot: QtBot, prior: list[GenKwConfig]
):
    state = ParameterConfigurationStateModel([_parameter("configured")])
    state.select_prior(prior)

    with (
        qtbot.assertNotEmitted(state.changed),
        pytest.raises(RuntimeError, match="cannot be changed"),
    ):
        state.apply_update_strategies({ParameterType.GEN_KW: LocalizationType.ADAPTIVE})

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

    state.apply_update_strategies({ParameterType.GEN_KW: LocalizationType.ADAPTIVE})

    assert [p.update_strategy for p in state.parameters] == [
        None,
        LocalizationType.ADAPTIVE,
    ]


def test_that_update_strategies_only_include_parameters_with_a_strategy():
    state = ParameterConfigurationStateModel(
        [_parameter("without", update_strategy=None), _parameter("with")]
    )

    assert state.update_strategies == {ParameterType.GEN_KW: LocalizationType.GLOBAL}


def test_that_update_strategies_are_keyed_by_parameter_type():
    state = ParameterConfigurationStateModel(
        [_parameter("gen_kw"), _surface("surface")]
    )

    assert all(isinstance(key, ParameterType) for key in state.update_strategies)
    assert set(state.update_strategies) == {
        ParameterType.GEN_KW,
        ParameterType.SURFACE,
    }


def test_that_update_strategies_reflect_applied_draft():
    state = ParameterConfigurationStateModel([_parameter("configured")])

    state.apply_update_strategies({ParameterType.GEN_KW: LocalizationType.ADAPTIVE})

    assert state.update_strategies == {ParameterType.GEN_KW: LocalizationType.ADAPTIVE}


def test_that_toggling_prior_switches_active_parameters_and_emits_changed(
    qtbot: QtBot,
):
    configured = [_parameter("configured")]
    prior = [_parameter("prior")]
    state = ParameterConfigurationStateModel(configured)

    with qtbot.waitSignal(state.changed, timeout=0):
        state.select_prior(prior)
    assert state.parameters == prior
    assert state.has_prior

    with qtbot.waitSignal(state.changed, timeout=0):
        state.deselect_prior()
    assert state.parameters == configured
    assert not state.has_prior


def test_that_deselecting_prior_restores_configured_draft():
    state = ParameterConfigurationStateModel([_parameter("configured")])
    state.apply_update_strategies({ParameterType.GEN_KW: LocalizationType.ADAPTIVE})

    state.select_prior([_parameter("prior")])
    state.deselect_prior()

    assert state.update_strategies == {ParameterType.GEN_KW: LocalizationType.ADAPTIVE}
