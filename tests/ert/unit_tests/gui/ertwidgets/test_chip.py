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


def test_that_a_long_ensemble_name_is_elided_and_given_in_full_in_the_tooltip(qtbot):
    chip = SourceChip()
    qtbot.addWidget(chip)
    long_name = "experiment_" + "x" * 200

    chip.set_source(DataSource.SELECTED_ENSEMBLE, long_name)

    assert chip.text() != f"From {long_name}"
    assert chip.toolTip() == f"From {long_name}"
    assert chip.width() < 400


def test_that_chip_uses_inverted_colors_in_dark_mode(qtbot, monkeypatch):
    monkeypatch.setattr("ert.gui.ertwidgets.chip.is_dark_mode", lambda: True)
    chip = SourceChip()
    qtbot.addWidget(chip)

    assert "color: white" in chip.styleSheet()
