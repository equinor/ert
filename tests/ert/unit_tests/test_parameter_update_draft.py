import subprocess
import sys
from textwrap import dedent

import pytest

from ert.config import GenKwConfig, LocalizationType
from ert.parameter_update_draft import ParameterUpdateDraft


def _parameter(name: str) -> GenKwConfig:
    return GenKwConfig(name=name, distribution={"name": "uniform", "min": 0, "max": 1})


def test_that_parameter_drafts_do_not_mutate_their_sources():
    source = _parameter("configured")
    draft = ParameterUpdateDraft([source])
    draft.apply_strategies_by_type({"gen_kw": LocalizationType.ADAPTIVE})

    assert draft.parameters[0].update_strategy == LocalizationType.ADAPTIVE
    assert source.update_strategy == LocalizationType.GLOBAL
    snapshot = draft.parameters
    snapshot[0].update_strategy = LocalizationType.GLOBAL
    assert draft.parameters[0].update_strategy == LocalizationType.ADAPTIVE


def test_that_leaving_a_prior_discards_its_edits_but_preserves_configured_edits():
    draft = ParameterUpdateDraft([_parameter("configured")])
    draft.apply_strategies_by_type({"gen_kw": LocalizationType.ADAPTIVE})
    prior = [_parameter("prior")]
    draft.select_prior("prior-id", prior)
    draft.apply_strategies_by_type({"gen_kw": LocalizationType.ADAPTIVE})
    draft.use_configured_parameters()
    assert draft.parameters[0].update_strategy == LocalizationType.ADAPTIVE
    draft.select_prior("prior-id", prior)
    assert draft.parameters[0].update_strategy == LocalizationType.GLOBAL


def test_that_reset_restores_both_parameter_drafts():
    draft = ParameterUpdateDraft([_parameter("configured")])
    draft.apply_strategies_by_type({"gen_kw": LocalizationType.ADAPTIVE})
    draft.select_prior("prior-id", [_parameter("prior")])
    draft.apply_strategies_by_type({"gen_kw": LocalizationType.ADAPTIVE})
    assert draft.reset_all_overrides()

    assert not draft.has_changes
    assert draft.overrides == {}
    draft.use_configured_parameters()
    assert draft.parameters[0].update_strategy == LocalizationType.GLOBAL


def test_that_refreshing_the_same_prior_preserves_its_draft():
    draft = ParameterUpdateDraft([])
    prior = [_parameter("prior")]
    draft.select_prior("prior-id", prior)
    draft.apply_strategies_by_type({"gen_kw": LocalizationType.ADAPTIVE})

    assert not draft.select_prior("prior-id", prior)
    assert draft.parameters[0].update_strategy == LocalizationType.ADAPTIVE


def test_that_missing_prior_never_falls_back_to_configured_parameters():
    draft = ParameterUpdateDraft([_parameter("configured")])
    draft.select_prior(None, ())

    assert not draft.has_parameter_source
    assert draft.parameters == []


@pytest.mark.parametrize("prior_id", ["first", "second"])
def test_that_changing_prior_identity_or_contents_discards_prior_overrides(prior_id):
    draft = ParameterUpdateDraft([])
    draft.select_prior("first", [_parameter("old")])
    draft.apply_strategies_by_type({"gen_kw": LocalizationType.ADAPTIVE})
    parameters = [_parameter("old" if prior_id == "second" else "new")]

    assert draft.select_prior(prior_id, parameters)
    assert draft.prior_id == prior_id
    assert draft.parameters == parameters
    assert draft.overrides == {}


def test_that_unchanged_operations_report_no_change():
    draft = ParameterUpdateDraft([_parameter("configured")])
    assert not draft.use_configured_parameters()
    assert not draft.reset_all_overrides()
    assert not draft.apply_strategies_by_type({"gen_kw": LocalizationType.GLOBAL})
    assert draft.select_prior(None, ())
    assert not draft.select_prior(None, ())
    assert draft.use_configured_parameters()
    assert draft.prior_id is None
    assert draft.has_parameter_source
    assert not draft.use_configured_parameters()


def test_that_restoring_a_baseline_strategy_removes_the_override():
    draft = ParameterUpdateDraft([_parameter("configured")])
    assert draft.apply_strategies_by_type({"gen_kw": LocalizationType.ADAPTIVE})
    assert draft.apply_strategies_by_type({"gen_kw": LocalizationType.GLOBAL})
    assert draft.overrides == {}
    assert not draft.has_changes
    assert not draft.reset_all_overrides()


def test_that_rejected_strategies_preserve_parameters_and_overrides():
    draft = ParameterUpdateDraft([_parameter("configured")])
    draft.apply_strategies_by_type({"gen_kw": LocalizationType.ADAPTIVE})
    parameters, overrides = draft.parameters, draft.overrides

    with pytest.raises(ValueError, match="Distance localization is not supported"):
        draft.apply_strategies_by_type({"gen_kw": LocalizationType.DISTANCE})

    assert draft.parameters == parameters
    assert draft.overrides == overrides


def test_that_an_empty_loaded_prior_is_distinct_from_an_unavailable_prior():
    draft = ParameterUpdateDraft([])
    assert draft.select_prior("empty", ())
    assert draft.has_parameter_source
    assert draft.parameters == []
    assert draft.select_prior(None, ())
    assert not draft.has_parameter_source
    assert draft.parameters == []


def test_that_prior_parameters_and_returned_overrides_are_isolated_from_the_draft():
    source = _parameter("prior")
    draft = ParameterUpdateDraft([])
    draft.select_prior("prior", [source])
    source.update_strategy = LocalizationType.ADAPTIVE
    assert draft.parameters[0].update_strategy == LocalizationType.GLOBAL
    draft.apply_strategies_by_type({"gen_kw": LocalizationType.ADAPTIVE})
    draft.overrides.clear()
    assert draft.overrides == {"prior": LocalizationType.ADAPTIVE}


@pytest.mark.slow
def test_that_the_draft_can_be_imported_and_used_without_qt():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            dedent("""\
                import importlib.abc
                import sys

                class RejectQt(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path=None, target=None):
                        if fullname.startswith(("PyQt", "PySide")):
                            raise AssertionError(f"Unexpected Qt import: {fullname}")

                sys.meta_path.insert(0, RejectQt())
                from ert.parameter_update_draft import ParameterUpdateDraft
                draft = ParameterUpdateDraft([])
                assert draft.has_parameter_source
                assert not draft.reset_all_overrides()
                assert not any(
                    name.startswith(("PyQt", "PySide")) for name in sys.modules
                )
                """),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
