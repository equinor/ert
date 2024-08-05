import pytest
from pytestqt.qtbot import QtBot
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QPlainTextEdit

from ert.gui.tools.search_bar import SearchBar


@pytest.mark.parametrize(
    "text, search",
    [
        pytest.param(
            "Testing search functionality.",
            "Test",
            id="Highlights matching text",
        ),
        pytest.param(
            "Testing special*@#$%characters.",
            "special*@#$%characters",
            id="Works with special characters",
        ),
        pytest.param(
            "Testing search functionality.",
            "new",
            id="Clears selection on search change",
        ),
        pytest.param("Testing search functionality.", "", id="Empty selection"),
    ],
)
def test_search_bar(text, search, qtbot: QtBot):
    text_box = QPlainTextEdit()
    text_box.setPlainText(text)
    search_bar = SearchBar(text_box)

    search_bar.select_text(0, 5)  # Simulate previous highlight

    search_bar.search_bar_changed(search)
    cursor = text_box.textCursor()

    if search and search in text:
        assert cursor.charFormat().background().color() == QColor("yellow")
        assert cursor.selectedText() == search
    else:
        assert cursor.charFormat().background().color() == QColor("white")
        assert not cursor.selectedText()
