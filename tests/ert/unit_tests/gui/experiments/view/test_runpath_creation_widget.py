from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PyQt6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from ert.gui.experiments.view.runpath_progress_widget import RunpathProgressWidget
from ert.run_models.run_model import RunModel


def test_that_widget_shows_correct_state_on_start(qtbot: QtBot) -> None:
    widget = RunpathProgressWidget(
        initial_status_text="Preparing runpaths...",
        completed_action="created",
    )
    qtbot.addWidget(widget)

    widget.start(10)

    assert widget._bar.maximum() == 10
    assert widget._bar.value() == 0
    assert widget._label.text() == "0 / 10 runpaths created"


def test_that_widget_updates_label_and_bar_on_progress_events(qtbot: QtBot) -> None:
    widget = RunpathProgressWidget(
        initial_status_text="Preparing runpaths...",
        completed_action="created",
    )
    qtbot.addWidget(widget)
    widget.start(4)

    widget.advance()
    assert widget._bar.value() == 1
    assert widget._label.text() == "1 / 4 runpaths created"

    widget.advance()
    widget.advance()
    assert widget._bar.value() == 3
    assert widget._label.text() == "3 / 4 runpaths created"


def test_that_start_resets_widget_state(qtbot: QtBot) -> None:
    widget = RunpathProgressWidget(
        initial_status_text="Preparing runpaths...",
        completed_action="created",
    )
    qtbot.addWidget(widget)

    widget.start(5)
    widget.advance()
    widget.advance()

    widget.start(2)

    assert widget._bar.maximum() == 2
    assert widget._bar.value() == 0
    assert widget._label.text() == "0 / 2 runpaths created"


def test_that_widget_tracks_delete_progress_during_runpath_removal(
    qtbot: QtBot, use_tmpdir
) -> None:
    removed_runpaths = []

    for iens in (0, 2):
        runpath = Path(f"Case_Name/realization-{iens}/iter-0")
        runpath.mkdir(parents=True)
        (runpath / "dummy").touch()
        removed_runpaths.append(runpath)

    widget = RunpathProgressWidget(
        initial_status_text="Deleting runpaths...",
        completed_action="deleted",
    )
    qtbot.addWidget(widget)
    widget.show()

    qtbot.waitUntil(widget.isVisible)
    assert widget._label.text() == "Deleting runpaths..."

    RunModel.rm_runpath(
        SimpleNamespace(paths=[str(runpath) for runpath in removed_runpaths]),
        progress_tracker=widget,
        progress_callback=QApplication.processEvents,
    )

    assert widget._bar.maximum() == 2
    assert widget._bar.value() == 2
    assert widget._label.text() == "2 / 2 runpaths deleted"
    assert all(not runpath.exists() for runpath in removed_runpaths)
