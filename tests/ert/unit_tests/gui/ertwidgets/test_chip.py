from PyQt6.QtCore import Qt

from ert.gui import eds_colors
from ert.gui.eds_theme import render_eds_stylesheet
from ert.gui.ertwidgets.chip import DataSource, SourceChip


def test_that_chip_shows_config_file_as_default_source(qtbot):
    chip = SourceChip()
    qtbot.addWidget(chip)

    assert chip.source == DataSource.CONFIG
    assert chip.text() == "From config file"
    assert not chip.toolTip()


def test_that_chip_shows_the_name_of_the_selected_ensemble(qtbot):
    chip = SourceChip()
    qtbot.addWidget(chip)

    chip.set_source(DataSource.SELECTED_ENSEMBLE, "experiment : iter-0")

    assert chip.source == DataSource.SELECTED_ENSEMBLE
    assert chip.text() == "From experiment : iter-0"


def test_that_chip_names_the_selected_ensemble_source_without_an_ensemble_name(qtbot):
    chip = SourceChip()
    qtbot.addWidget(chip)

    chip.set_source(DataSource.SELECTED_ENSEMBLE)

    assert chip.text() == "From selected ensemble"


def test_that_chip_shows_html_in_an_ensemble_name_literally(qtbot):
    chip = SourceChip()
    qtbot.addWidget(chip)

    chip.set_source(DataSource.SELECTED_ENSEMBLE, "<b>case")

    assert chip.textFormat() == Qt.TextFormat.PlainText
    assert chip.text() == "From <b>case"


def test_that_a_long_ensemble_name_is_elided_and_given_in_full_in_the_tooltip(qtbot):
    chip = SourceChip()
    qtbot.addWidget(chip)
    long_name = "experiment_" + "x" * 200

    chip.set_source(DataSource.SELECTED_ENSEMBLE, long_name)

    assert chip.text() != f"From {long_name}"
    assert chip.toolTip() == f"From {long_name}"
    assert chip.sizeHint().width() < 400


def _chip_background(qtbot, qapp, source):
    qapp.setStyleSheet(render_eds_stylesheet())
    try:
        chip = SourceChip()
        qtbot.addWidget(chip)
        chip.set_source(source)
        chip.show()
        return chip.grab().toImage().pixelColor(chip.width() // 2, 1).name()
    finally:
        qapp.setStyleSheet("")


def test_that_chip_background_comes_from_the_eds_stylesheet_for_each_source(
    qtbot, qapp
):
    assert _chip_background(qtbot, qapp, DataSource.CONFIG) == (
        eds_colors.BACKGROUND_ACCENT_MUTED
    )
    assert _chip_background(qtbot, qapp, DataSource.SELECTED_ENSEMBLE) == (
        eds_colors.BACKGROUND_WARNING_MUTED
    )


def test_that_dark_stylesheet_gives_the_chip_dark_tokens(monkeypatch):
    monkeypatch.setattr("ert.gui.eds_theme.is_dark_mode", lambda: True)

    stylesheet = render_eds_stylesheet()

    dark_background = eds_colors.DARK_COLORS["BACKGROUND_ACCENT_MUTED"]
    assert f"background-color: {dark_background}" in stylesheet
    assert eds_colors.BACKGROUND_ACCENT_MUTED not in stylesheet
