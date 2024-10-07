from ert.gui.tools.plot.plottery import PlotConfig, PlotConfigHistory


def test_plot_config_history():
    test_pc = PlotConfig(title="test_1")
    history = PlotConfigHistory("test", test_pc)

    assert history.getPlotConfig().title() == test_pc.title()
    assert history.getPlotConfig() != test_pc

    assert not history.isUndoPossible()
    assert not history.isRedoPossible()

    history.applyChanges(PlotConfig(title="test_2"))
    assert history.isUndoPossible()
    assert not history.isRedoPossible()
    assert history.getPlotConfig().title() == "test_2"

    history.undoChanges()
    assert not history.isUndoPossible()
    assert history.isRedoPossible()
    assert history.getPlotConfig().title() == "test_1"

    history.redoChanges()
    assert history.isUndoPossible()
    assert not history.isRedoPossible()
    assert history.getPlotConfig().title() == "test_2"

    history.resetChanges()
    assert history.isUndoPossible()
    assert not history.isRedoPossible()
    assert history.getPlotConfig().title() == "test_1"

    history.undoChanges()
    assert history.isUndoPossible()
    assert history.isRedoPossible()
    assert history.getPlotConfig().title() == "test_2"
