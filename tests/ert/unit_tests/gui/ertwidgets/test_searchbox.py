from PyQt6.QtTest import QSignalSpy

from ert.gui.ertwidgets.searchbox import SearchBox


def test_search_box_does_not_emit_filter_changed_when_focused(qtbot):
    search_box = SearchBox(debounce_timeout=10)
    qtbot.addWidget(search_box)

    signal_spy = QSignalSpy(search_box.filterChanged)

    search_box.setFocus()
    qtbot.wait(20)

    assert len(signal_spy) == 0
    assert not search_box._search_pending
    assert not search_box._pending_indicator.isVisible()


def test_search_box_shows_pending_indicator_while_debounce_is_active(qtbot):
    search_box = SearchBox(debounce_timeout=100)
    qtbot.addWidget(search_box)
    search_box.show()

    search_box.activateSearch()
    qtbot.keyClicks(search_box, "abc")

    assert search_box.filter() == "abc"
    assert search_box._search_pending
    assert search_box._pending_indicator.isVisible()

    with qtbot.waitSignal(search_box.filterChanged, timeout=200) as blocker:
        pass

    assert blocker.args == ["abc"]
    assert not search_box._search_pending
    assert not search_box._pending_indicator.isVisible()
