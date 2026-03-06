from __future__ import annotations

from pytestqt.qtbot import QtBot

from ert.gui.experiments.view.runpath_creation_widget import RunpathCreationProgressBar
from ert.run_models.event import RunPathCreationEvent


def _start(total: int = 5) -> RunPathCreationEvent:
    return RunPathCreationEvent(
        sub_type="StartingTotalRunPathCreation",
        total_runpaths_to_create=total,
    )


def _update(n: int, total: int = 5) -> RunPathCreationEvent:
    return RunPathCreationEvent(
        sub_type="TotalRunPathCreationUpdate",
        runpath_number=n,
        total_runpaths_to_create=total,
    )


def _finished() -> RunPathCreationEvent:
    return RunPathCreationEvent(sub_type="FinishedTotalRunPathCreation")


def test_that_widget_shows_correct_state_on_start(qtbot: QtBot) -> None:
    widget = RunpathCreationProgressBar()
    qtbot.addWidget(widget)

    widget.handle_event(_start(total=10))

    assert widget._bar.maximum() == 10
    assert widget._bar.value() == 0
    assert "0" in widget._label.text()
    assert "10" in widget._label.text()


def test_that_widget_updates_label_and_bar_on_progress_events(qtbot: QtBot) -> None:
    widget = RunpathCreationProgressBar()
    qtbot.addWidget(widget)
    widget.handle_event(_start(total=4))

    widget.handle_event(_update(1, total=4))
    assert widget._bar.value() == 1
    assert "1" in widget._label.text()

    widget.handle_event(_update(3, total=4))
    assert widget._bar.value() == 3
    assert "3" in widget._label.text()


def test_that_finish_event_is_ignored_by_widget(qtbot: QtBot) -> None:
    widget = RunpathCreationProgressBar()
    qtbot.addWidget(widget)
    widget.handle_event(_start())
    widget.handle_event(_finished())
