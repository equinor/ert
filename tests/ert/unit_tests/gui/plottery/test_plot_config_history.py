from ert.gui.tools.plot.plottery import PlotConfig, PlotConfigHistory


def test_plot_config_history():
    test_pc = PlotConfig(title="test_1")
    history = PlotConfigHistory("test", test_pc)

    assert history.get_plot_config().title() == test_pc.title()
    assert history.get_plot_config() != test_pc

    assert not history.is_undo_possible()
    assert not history.is_redo_possible()

    history.apply_changes(PlotConfig(title="test_2"))
    assert history.is_undo_possible()
    assert not history.is_redo_possible()
    assert history.get_plot_config().title() == "test_2"

    history.undo_changes()
    assert not history.is_undo_possible()
    assert history.is_redo_possible()
    assert history.get_plot_config().title() == "test_1"

    history.redo_changes()
    assert history.is_undo_possible()
    assert not history.is_redo_possible()
    assert history.get_plot_config().title() == "test_2"

    history.reset_changes()
    assert history.is_undo_possible()
    assert not history.is_redo_possible()
    assert history.get_plot_config().title() == "test_1"

    history.undo_changes()
    assert history.is_undo_possible()
    assert history.is_redo_possible()
    assert history.get_plot_config().title() == "test_2"
