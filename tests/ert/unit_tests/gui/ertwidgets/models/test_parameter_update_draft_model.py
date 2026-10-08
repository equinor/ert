import pytest

from ert.config import GenKwConfig, LocalizationType
from ert.gui.ertwidgets.models.parameter_update_draft_model import (
    ParameterUpdateDraftModel,
)


def _parameter() -> GenKwConfig:
    return GenKwConfig(
        name="configured",
        distribution={"name": "uniform", "min": 0, "max": 1},
    )


def test_that_draft_model_notifies_observers_when_the_draft_changes():
    model = ParameterUpdateDraftModel([_parameter()])
    changes: list[None] = []
    model.changed.connect(lambda: changes.append(None))

    model.apply_strategies_by_type({"gen_kw": LocalizationType.ADAPTIVE})

    assert model.revision == 1
    assert changes == [None]


def test_that_draft_model_does_not_notify_observers_for_unchanged_drafts():
    model = ParameterUpdateDraftModel([_parameter()])
    changes: list[None] = []
    model.changed.connect(lambda: changes.append(None))

    model.apply_strategies_by_type({"gen_kw": LocalizationType.GLOBAL})

    assert model.revision == 0
    assert changes == []


def test_that_source_changes_and_resets_notify_once_after_updating_the_revision():
    model = ParameterUpdateDraftModel([_parameter()])
    revisions: list[int] = []
    model.changed.connect(lambda: revisions.append(model.revision))

    model.select_prior("prior", [_parameter()])
    assert model.prior_id == "prior"
    assert model.has_parameter_source
    model.select_prior("prior", [_parameter()])
    assert revisions == [1]
    model.apply_strategies_by_type({"gen_kw": LocalizationType.ADAPTIVE})
    model.reset_all_overrides()
    assert revisions == [1, 2, 3]
    assert not model.has_changes
    model.reset_all_overrides()
    assert revisions == [1, 2, 3]
    model.use_configured_parameters()
    assert model.prior_id is None
    model.use_configured_parameters()
    assert revisions == [1, 2, 3, 4]
    model.select_prior(None, ())
    assert not model.has_parameter_source
    assert model.parameters == []
    model.select_prior(None, ())
    assert revisions == [1, 2, 3, 4, 5]


def test_that_rejected_strategies_do_not_change_the_model_or_notify_observers():
    model = ParameterUpdateDraftModel([_parameter()])
    model.apply_strategies_by_type({"gen_kw": LocalizationType.ADAPTIVE})
    parameters, overrides, revision = model.parameters, model.overrides, model.revision
    revisions: list[int] = []
    model.changed.connect(lambda: revisions.append(model.revision))

    with pytest.raises(ValueError, match="Distance localization is not supported"):
        model.apply_strategies_by_type({"gen_kw": LocalizationType.DISTANCE})

    assert model.parameters == parameters
    assert model.overrides == overrides
    assert model.revision == revision
    assert revisions == []


def test_that_revision_cannot_be_assigned_by_consumers():
    model = ParameterUpdateDraftModel([])
    with pytest.raises(AttributeError):
        model.revision = 10
    assert model.revision == 0
