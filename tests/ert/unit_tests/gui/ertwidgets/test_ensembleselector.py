import pytest

from ert.config import GenDataConfig, GenKwConfig
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import EnsembleSelector
from ert.storage.realization_storage_state import RealizationStorageState


@pytest.fixture
def uniform_parameter():
    return GenKwConfig(
        name="KEY_1",
        distribution={"name": "uniform", "min": 0, "max": 1},
    )


@pytest.fixture
def response():
    return GenDataConfig(keys=["response"])


@pytest.fixture
def notifier():
    return ErtNotifier()


@pytest.fixture
def storage_with_three_ensembles(storage, notifier):
    ensemble_a = storage.create_experiment().create_ensemble(name="a", ensemble_size=1)
    ensemble_b = storage.create_experiment().create_ensemble(
        name="b", ensemble_size=1, prior_ensemble=ensemble_a
    )
    storage.create_experiment().create_ensemble(
        name="c", ensemble_size=1, prior_ensemble=ensemble_b
    )

    notifier.set_storage(str(storage.path))


def test_that_ensemble_selector_is_empty_when_no_storage_is_set(qtbot, notifier):
    widget = EnsembleSelector(notifier)
    qtbot.addWidget(widget)

    assert widget.count() == 0


def test_that_ensemble_selector_is_populated_regardless_of_storage_creation_order(
    qtbot, notifier, storage
):
    ensemble = storage.create_experiment().create_ensemble(
        name="default", ensemble_size=1
    )

    # Adding a storage after widget creation populates it
    widget = EnsembleSelector(notifier)
    qtbot.addWidget(widget)
    assert widget.count() == 0

    notifier.set_storage(str(storage.path))
    assert widget.count() == 1
    assert widget.currentData() == str(ensemble.id)

    # Creating EnsembleSelector after storage has been created populates it
    widget = EnsembleSelector(notifier)
    qtbot.addWidget(widget)
    assert widget.count() == 1
    assert widget.currentData() == str(ensemble.id)


def test_that_changing_one_selector_leaves_other_selectors_and_notifier_unchanged(
    qtbot, notifier, storage
):
    ensemble_a = storage.create_experiment().create_ensemble(
        name="default_a", ensemble_size=1
    )
    ensemble_b = storage.create_experiment().create_ensemble(
        name="default_b", ensemble_size=1
    )
    notifier.set_storage(str(storage.path))

    widget_a = EnsembleSelector(notifier)
    widget_b = EnsembleSelector(notifier)
    qtbot.addWidget(widget_a)
    qtbot.addWidget(widget_b)

    widget_b_selection = widget_b.selected_ensemble
    notifier_selection = notifier.current_ensemble

    # Select the ensemble in widget_a that is not currently selected
    assert widget_a.selected_ensemble is not None
    other = ensemble_a if widget_a.selected_ensemble.id == ensemble_b.id else ensemble_b
    widget_a.setCurrentIndex(widget_a.findData(str(other.id)))

    assert widget_a.selected_ensemble is not None
    assert widget_a.selected_ensemble.id == other.id
    assert widget_b.selected_ensemble == widget_b_selection
    assert notifier.current_ensemble == notifier_selection


def test_ensembles_are_sorted_failed_first_then_by_start_time(storage):
    ensemble_a = storage.create_experiment().create_ensemble(
        name="default_a", ensemble_size=1
    )
    ensemble_b = storage.create_experiment().create_ensemble(
        name="default_b", ensemble_size=1
    )
    ensemble_c = storage.create_experiment().create_ensemble(
        name="default_a", ensemble_size=1
    )
    ensemble_b.set_failure(0, RealizationStorageState.FAILURE_IN_CURRENT)
    assert EnsembleSelector.sort_ensembles([ensemble_a, ensemble_b, ensemble_c]) == [
        ensemble_b,
        ensemble_c,
        ensemble_a,
    ]


def test_that_when_filters_are_not_provided_then_all_ensembles_are_selected(
    qtbot, notifier, storage_with_three_ensembles
):
    widget = EnsembleSelector(notifier)
    qtbot.addWidget(widget)
    assert widget.count() == 3


def test_that_when_filters_are_empty_then_all_ensembles_are_selected(
    qtbot, notifier, storage_with_three_ensembles
):
    widget = EnsembleSelector(notifier, filters=[])
    qtbot.addWidget(widget)
    assert widget.count() == 3


def test_that_when_filters_are_independent_then_all_fitting_ensembles_are_selected(
    qtbot, notifier, storage_with_three_ensembles
):
    widget = EnsembleSelector(
        notifier,
        filters=[
            lambda ensembles: (e for e in ensembles if e.name == "b"),
            lambda ensembles: (e for e in ensembles if e.name == "c"),
        ],
    )
    qtbot.addWidget(widget)
    assert widget.count() == 2


def test_that_when_filters_are_interdependent_then_ensembles_are_not_duplicated(
    qtbot, notifier, storage_with_three_ensembles
):
    widget = EnsembleSelector(
        notifier,
        filters=[
            lambda ensembles: (e for e in ensembles if e.name == "a"),
            lambda ensembles: (e for e in ensembles if e.parent is None),
        ],
    )
    qtbot.addWidget(widget)
    assert widget.count() == 1
