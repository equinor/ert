import pytest

from ert._c_wrappers.enkf.ert_config import ErtConfig
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets.caseselector import CaseSelector


@pytest.fixture
def notifier():
    return ErtNotifier(ErtConfig())


def test_empty(qtbot, notifier):
    widget = CaseSelector(notifier)
    qtbot.addWidget(widget)

    assert widget.count() == 0


def test_current_case(qtbot, notifier, storage):
    ensemble = storage.create_experiment().create_ensemble(
        name="default", ensemble_size=1
    )

    # Adding a storage after widget creation populates it
    widget = CaseSelector(notifier)
    qtbot.addWidget(widget)
    assert widget.count() == 0

    notifier.set_storage(storage)
    notifier.set_current_case(ensemble)
    assert widget.count() == 1
    assert widget.currentData() == ensemble

    # Creating CaseSelector after storage has been created populates it
    widget = CaseSelector(notifier)
    qtbot.addWidget(widget)
    assert widget.count() == 1
    assert widget.currentData() == ensemble


def test_changing_case(qtbot, notifier, storage):
    ensemble_a = storage.create_experiment().create_ensemble(
        name="default_a", ensemble_size=1
    )
    ensemble_b = storage.create_experiment().create_ensemble(
        name="default_b", ensemble_size=1
    )

    notifier.set_storage(storage)
    notifier.set_current_case(ensemble_a)
    widget_a = CaseSelector(notifier)
    widget_b = CaseSelector(notifier)
    qtbot.addWidget(widget_a)
    qtbot.addWidget(widget_b)

    assert widget_a.count() == 2
    assert widget_b.count() == 2

    # First ensemble is selected in both
    assert widget_a.currentData() == ensemble_a
    assert widget_b.currentData() == ensemble_a

    # Second ensemble is selected via signal, changing both widgets'
    # selections
    notifier.set_current_case(ensemble_b)
    assert widget_a.currentData() == ensemble_b
    assert widget_b.currentData() == ensemble_b

    # Changing back to first ensemble via widget sets the global current_case
    qtbot.keyClicks(
        widget_a,
        widget_a.itemText(widget_a.findData(ensemble_a)),
    )
    assert notifier.current_case == ensemble_a
    assert widget_a.currentData() == ensemble_a
    assert widget_b.currentData() == ensemble_a
