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
    ("flag", "expected"),
    [
        (True, ["my-experiment : child"]),
        (False, ["my-experiment : child", "my-experiment : parent"]),
    ],
)
def test_show_only_no_parent(
    qtbot, notifier, storage, uniform_parameter, response, flag, expected
):
    experiment = storage.create_experiment(
        experiment_config={
            "parameter_configuration": [uniform_parameter.model_dump(mode="json")],
            "response_configuration": [response.model_dump(mode="json")],
        },
        name="my-experiment",
    )
    ensemble = experiment.create_ensemble(name="parent", ensemble_size=1)
    experiment.create_ensemble(name="child", ensemble_size=1, prior_ensemble=ensemble)

    notifier.set_storage(str(storage.path))
    notifier.set_current_ensemble_id(ensemble.id)

    widget = EnsembleSelector(notifier, show_only_no_children=flag)
    qtbot.addWidget(widget)
    assert [widget.itemText(i) for i in range(widget.count())] == expected
