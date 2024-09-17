import pytest

from ert.config import ErtConfig, GenDataConfig, GenKwConfig
from ert.config.gen_kw_config import TransformFunctionDefinition
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets.ensembleselector import EnsembleSelector


@pytest.fixture
def uniform_parameter():
    return GenKwConfig(
        name="parameter",
        forward_init=False,
        template_file="",
        transform_function_definitions=[
            TransformFunctionDefinition("KEY1", "UNIFORM", [0, 1]),
        ],
        output_file="kw.txt",
        update=True,
    )


@pytest.fixture
def response():
    return GenDataConfig(keys=["response"])


@pytest.fixture
def notifier():
    return ErtNotifier(ErtConfig())


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

    notifier.set_storage(storage)
    notifier.set_current_ensemble(ensemble)
    assert widget.count() == 1
    assert widget.currentData() == ensemble

    # Creating EnsembleSelector after storage has been created populates it
    widget = EnsembleSelector(notifier)
    qtbot.addWidget(widget)
    assert widget.count() == 1
    assert widget.currentData() == ensemble


def test_changing_ensemble(qtbot, notifier, storage):
    ensemble_a = storage.create_experiment().create_ensemble(
        name="default_a", ensemble_size=1
    )
    ensemble_b = storage.create_experiment().create_ensemble(
        name="default_b", ensemble_size=1
    )

    notifier.set_storage(storage)
    notifier.set_current_ensemble(ensemble_a)
    widget_a = EnsembleSelector(notifier)
    widget_b = EnsembleSelector(notifier)
    qtbot.addWidget(widget_a)
    qtbot.addWidget(widget_b)

    assert widget_a.count() == 2
    assert widget_b.count() == 2

    # First ensemble is selected in both
    assert widget_a.currentData() == ensemble_a
    assert widget_b.currentData() == ensemble_a

    # Second ensemble is selected via signal, changing both widgets'
    # selections
    notifier.set_current_ensemble(ensemble_b)
    assert widget_a.currentData() == ensemble_b
    assert widget_b.currentData() == ensemble_b

    # Changing back to first ensemble via widget sets the global current_ensemble
    qtbot.keyClicks(
        widget_a,
        widget_a.itemText(widget_a.findData(ensemble_a)),
    )
    assert notifier.current_ensemble == ensemble_a
    assert widget_a.currentData() == ensemble_a
    assert widget_b.currentData() == ensemble_a


@pytest.mark.parametrize(
    "flag, expected",
    [
        (True, ["my-experiment : child"]),
        (False, ["my-experiment : child", "my-experiment : parent"]),
    ],
)
def test_show_only_no_parent(
    qtbot, notifier, storage, uniform_parameter, response, flag, expected
):
    experiment = storage.create_experiment(
        parameters=[uniform_parameter], responses=[response], name="my-experiment"
    )
    ensemble = experiment.create_ensemble(name="parent", ensemble_size=1)
    experiment.create_ensemble(name="child", ensemble_size=1, prior_ensemble=ensemble)

    notifier.set_storage(storage)
    notifier.set_current_ensemble(ensemble)

    widget = EnsembleSelector(notifier, show_only_no_children=flag)
    qtbot.addWidget(widget)
    assert [widget.itemText(i) for i in range(widget.count())] == expected
