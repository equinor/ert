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


def test_empty(qtbot, notifier):
    widget = EnsembleSelector(notifier)
    qtbot.addWidget(widget)

    assert widget.count() == 0


def test_current_ensemble(qtbot, notifier, storage):
    ensemble = storage.create_experiment().create_ensemble(
        name="default", ensemble_size=1
    )

    # Adding a storage after widget creation populates it
    widget = EnsembleSelector(notifier)
    qtbot.addWidget(widget)
    assert widget.count() == 0

    notifier.set_storage(str(storage.path))
    notifier.set_current_ensemble_id(ensemble.id)
    assert widget.count() == 1
    assert widget.currentData() == str(ensemble.id)

    # Creating EnsembleSelector after storage has been created populates it
    widget = EnsembleSelector(notifier)
    qtbot.addWidget(widget)
    assert widget.count() == 1
    assert widget.currentData() == str(ensemble.id)


def test_changing_ensemble(qtbot, notifier, storage):
    ensemble_a = storage.create_experiment().create_ensemble(
        name="default_a", ensemble_size=1
    )
    ensemble_b = storage.create_experiment().create_ensemble(
        name="default_b", ensemble_size=1
    )

    notifier.set_storage(str(storage.path))
    notifier.set_current_ensemble_id(ensemble_a.id)
    widget_a = EnsembleSelector(notifier)
    widget_b = EnsembleSelector(notifier)
    qtbot.addWidget(widget_a)
    qtbot.addWidget(widget_b)

    assert widget_a.count() == 2
    assert widget_b.count() == 2

    # Latest ensemble is selected in both
    assert widget_a.currentData() == str(ensemble_b.id)
    assert widget_b.currentData() == str(ensemble_b.id)

    # Second ensemble is selected via signal, widgets do not change'
    # selections
    notifier.set_current_ensemble_id(ensemble_b.id)
    assert widget_a.currentData() == str(ensemble_b.id)
    assert widget_b.currentData() == str(ensemble_b.id)

    # Changing back to first ensemble via widget sets the global current_ensemble
    qtbot.keyClicks(
        widget_a,
        widget_a.itemText(widget_a.findData(str(ensemble_a.id))),
    )
    assert notifier.current_ensemble.id == ensemble_a.id
    assert widget_a.currentData() == str(ensemble_a.id)
    assert widget_b.currentData() == str(ensemble_a.id)


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


@pytest.mark.parametrize(
    ("filters", "expected_ensembles_count"),
    [
        pytest.param(
            None,
            3,
            id="filters are None",
        ),
        pytest.param(
            [],
            3,
            id="filters are empty",
        ),
        pytest.param(
            [
                lambda ensembles: (e for e in ensembles if e.name == "b"),
                lambda ensembles: (e for e in ensembles if e.name == "c"),
            ],
            2,
            id="independent filters",
        ),
        pytest.param(
            [
                lambda ensembles: (e for e in ensembles if e.name == "a"),
                lambda ensembles: (e for e in ensembles if e.parent is None),
            ],
            1,
            id="duplicating filters",
        ),
    ],
)
def test_that_filters_are_applied(
    qtbot, notifier, storage, filters, expected_ensembles_count
):
    ensemble_a = storage.create_experiment().create_ensemble(name="a", ensemble_size=1)
    ensemble_b = storage.create_experiment().create_ensemble(
        name="b", ensemble_size=1, prior_ensemble=ensemble_a
    )
    storage.create_experiment().create_ensemble(
        name="c", ensemble_size=1, prior_ensemble=ensemble_b
    )

    notifier.set_storage(str(storage.path))

    widget = EnsembleSelector(notifier, filters=filters)
    qtbot.addWidget(widget)
    assert widget.count() == expected_ensembles_count
