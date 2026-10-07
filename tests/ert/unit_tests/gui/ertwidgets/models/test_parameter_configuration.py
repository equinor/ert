from ert.config import GenKwConfig, LocalizationType
from ert.gui.ertwidgets.models.parameter_configuration import ParameterConfiguration


def _parameter(name: str) -> GenKwConfig:
    return GenKwConfig(name=name, distribution={"name": "uniform", "min": 0, "max": 1})


def test_that_parameter_drafts_do_not_mutate_their_sources():
    source = _parameter("configured")
    state = ParameterConfiguration([source])
    state.apply_strategies({"gen_kw": LocalizationType.ADAPTIVE})

    assert state.parameters[0].update_strategy == LocalizationType.ADAPTIVE
    assert source.update_strategy == LocalizationType.GLOBAL
    snapshot = state.parameters
    snapshot[0].update_strategy = LocalizationType.GLOBAL
    assert state.parameters[0].update_strategy == LocalizationType.ADAPTIVE


def test_that_leaving_a_prior_discards_its_edits_but_preserves_configured_edits():
    state = ParameterConfiguration([_parameter("configured")])
    state.apply_strategies({"gen_kw": LocalizationType.ADAPTIVE})
    prior = [_parameter("prior")]
    state.select_prior(True, "prior-id", prior)
    state.apply_strategies({"gen_kw": LocalizationType.ADAPTIVE})
    state.select_prior(False)
    assert state.parameters[0].update_strategy == LocalizationType.ADAPTIVE
    state.select_prior(True, "prior-id", prior)
    assert state.parameters[0].update_strategy == LocalizationType.GLOBAL


def test_that_reset_restores_both_parameter_drafts():
    state = ParameterConfiguration([_parameter("configured")])
    state.apply_strategies({"gen_kw": LocalizationType.ADAPTIVE})
    state.select_prior(True, "prior-id", [_parameter("prior")])
    state.apply_strategies({"gen_kw": LocalizationType.ADAPTIVE})
    state.reset()

    assert not state.has_changes
    assert state.overrides == {}
    state.select_prior(False)
    assert state.parameters[0].update_strategy == LocalizationType.GLOBAL


def test_that_refreshing_the_same_prior_preserves_its_draft():
    state = ParameterConfiguration([])
    prior = [_parameter("prior")]
    state.select_prior(True, "prior-id", prior)
    state.apply_strategies({"gen_kw": LocalizationType.ADAPTIVE})
    revision = state.revision
    state.select_prior(True, "prior-id", prior)
    assert state.revision == revision
    assert state.parameters[0].update_strategy == LocalizationType.ADAPTIVE


def test_that_missing_prior_never_falls_back_to_configured_parameters():
    state = ParameterConfiguration([_parameter("configured")])
    state.select_prior(True)
    assert not state.available
    assert state.parameters == []
