import pytest
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QPlainTextEdit
from pytestqt.qtbot import QtBot

from ert.gui.tools.search_bar import SearchBar


@pytest.mark.parametrize(
    ("text", "search"),
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
        pytest.param("Testing search functionality.", " \b", id="Empty selection"),
        # Note, need a non-empty string to test
    ],
)
def test_search_bar_find(text, search, qtbot: QtBot):
    text_box = QPlainTextEdit()
    text_box.setPlainText(text)
    search_bar = SearchBar(text_box)

    search_bar.setText(search)
    cursor = text_box.textCursor()

    if search and search in text:
        assert cursor.selectedText() == search
    else:
        assert cursor.charFormat().background().color() == QColor("white")
        assert not cursor.selectedText()


def test_search_bar_find_next(qtbot: QtBot):
    text_box = QPlainTextEdit()
    text_box.setPlainText("Testing search functionality. Testing is fun.")
    search_bar = SearchBar(text_box)

    search_and_cursor_position = [
        ("Test", 4),  # First "Test"
        ("Test", 34),  # Second "Test"
        ("Test", 4),  # Wrap around to first "Test"
    ]

    for search_text, expected_position in search_and_cursor_position:
        search_bar.setText(search_text)
        assert text_box.textCursor().selectedText() == search_text
        assert text_box.textCursor().position() == expected_position
        search_bar._find_next_button.click()


def test_search_bar_highlight_all(qtbot: QtBot):
    text_box = QPlainTextEdit()

    # This string has GUI starting at position 8 and 30
    # The higlight of yellow (correctly) starts after the character, so from position 9
    input_text = "Testing GUI is so fun. I love GUI forever"
    text_box.setPlainText(input_text)
    search_bar = SearchBar(text_box)
    search_bar.setText("GUI")
    search_bar._highlight_all_button.click()
    cursor = text_box.textCursor()
    assert not cursor.selectedText()  # No text selected when highlighting

    search_hit_positions = (8, 9, 10, 30, 31, 32)
    for i in range(len(input_text)):
        cursor.setPosition(i)
        if (
            i - 1 in search_hit_positions
        ):  # -1 because the color apparently starts from the next character
            assert cursor.charFormat().background().color() == QColor("yellow")
        else:
            assert cursor.charFormat().background().color() == QColor("white")
