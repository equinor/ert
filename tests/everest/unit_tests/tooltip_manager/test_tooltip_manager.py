from matplotlib import pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent

from ert.gui.plotting.utils.plot_context import PlotType
from ert.gui.plotting.utils.plot_tools import PlotTools


def test_that_tooltip_manager_catches_wrong_data_type_for_the_plot_type_selected(
    plot_data_1D, caplog
):
    caplog.set_level("WARNING")
    x_data, labels = plot_data_1D

    fig, ax = plt.subplots()
    scatter = ax.scatter(x_data, x_data)

    PlotTools.labels_on_hover(
        PlotType.LINE,
        ax,
        fig,
        data=scatter,
        labels=labels,
    )

    assert len(caplog.records) == 1
    assert "Invalid data type PathCollection for line plot" in caplog.records[0].message


def test_that_scatter_plot_tooltip_manager_catches_different_num_labels_vs_datapoints(
    plot_data_1D, caplog
):
    caplog.set_level("WARNING")
    x_data, labels = plot_data_1D

    fig, ax = plt.subplots()
    scatter = ax.scatter(x_data, x_data)

    labels.append("Extra label")

    PlotTools.labels_on_hover(
        PlotType.SCATTER,
        ax,
        fig,
        data=scatter,
        labels=labels,
    )

    assert len(caplog.records) == 1
    assert "Found 11 labels and 10 points." in caplog.records[0].message


def test_that_bar_plot_tooltip_manager_catches_different_num_labels_vs_datapoints(
    plot_data_1D, caplog
):
    caplog.set_level("WARNING")
    x_data, labels = plot_data_1D

    fig, ax = plt.subplots()
    bars = ax.bar(x_data, x_data)

    labels.append("Extra label")

    PlotTools.labels_on_hover(
        PlotType.BAR,
        ax,
        fig,
        data=[bars],
        labels=labels,
    )

    assert len(caplog.records) == 1
    assert "Found 11 labels and 10 bars." in caplog.records[0].message


def test_that_line_plot_tooltip_manager_catches_different_num_labels_vs_datapoints(
    plot_data_2D, caplog
):
    caplog.set_level("WARNING")
    x_data, labels = plot_data_2D

    fig, ax = plt.subplots()

    lines = [ax.plot(x) for x in x_data]

    labels.append("Extra label")

    PlotTools.labels_on_hover(
        PlotType.LINE,
        ax,
        fig,
        data=lines,
        labels=labels,
    )

    assert len(caplog.records) == 1
    assert "Found 11 labels and 10 lines." in caplog.records[0].message


def test_that_tooltip_manager_catches_incorrect_event_type(plot_data_1D, caplog):
    caplog.set_level("WARNING")
    x_data, labels = plot_data_1D

    fig, ax = plt.subplots()
    scatter = ax.scatter(x_data, x_data)

    PlotTools.labels_on_hover(
        PlotType.SCATTER,
        ax,
        fig,
        data=scatter,
        labels=labels,
    )

    event = KeyEvent("test", ax.figure.canvas, key="a")
    ax.figure.canvas.callbacks.process("motion_notify_event", event)
    assert len(caplog.records) == 1
    assert (
        f"Expected a MouseEvent, got {type(event).__name__}"
        in caplog.records[0].message
    )

    ax.figure.canvas.callbacks.process("motion_notify_event", event)
    assert len(caplog.records) == 1, (
        "No new warning should be logged for repeated errors"
    )


def test_that_tooltip_manager_catches_invalid_axes(plot_data_1D, caplog):
    caplog.set_level("WARNING")
    x_data, labels = plot_data_1D

    fig, ax = plt.subplots()
    scatter = ax.scatter(x_data, x_data)

    PlotTools.labels_on_hover(
        PlotType.SCATTER,
        ax,
        fig,
        data=scatter,
        labels=labels,
    )

    event = MouseEvent("test", ax.figure.canvas, 0, 0)

    event.inaxes = None
    ax.figure.canvas.callbacks.process("motion_notify_event", event)

    assert len(caplog.records) == 0, (
        "No warning should be logged for events outside of axes"
    )
