from collections.abc import Callable
from datetime import UTC, datetime
from unittest.mock import MagicMock

import matplotlib.dates as mdates
import pandas as pd
import pytest
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from PyQt6.QtWidgets import (
    QComboBox,
    QDateTimeEdit,
    QDialog,
    QDialogButtonBox,
    QLineEdit,
)
from pytestqt.qtbot import QtBot

from ert.gui.plotting.utils import PlotConfig, PlotContext
from ert.gui.plotting.widgets.axis_limits_dialog import AxisLimitsDialog
from ert.gui.plotting.widgets.plot_widget import PlotWidget


def _create_plot_widget(qtbot: QtBot, y_values: list[float]) -> PlotWidget:
    def _plot(figure: Figure, *_args, **_kwargs) -> None:
        axes = figure.add_subplot(111)
        axes.plot(range(len(y_values)), y_values)
        axes.set_xlabel("X label")
        axes.set_ylabel("Y label")

    return _create_plot_widget_with(qtbot, _plot)


def _create_categorical_plot_widget(
    qtbot: QtBot, labels: list[list[str]]
) -> PlotWidget:
    """The x-axis has one category per label, as in the misfit plot. Each redraw
    uses the next list of labels.
    """
    labels_per_draw = iter(labels)

    def _plot(figure: Figure, *_args, **_kwargs) -> None:
        draw_labels = next(labels_per_draw)
        axes = figure.add_subplot(111)
        axes.plot(range(len(draw_labels)), range(len(draw_labels)))
        axes.set_xlim(-0.5, len(draw_labels) - 0.5)
        axes.set_xticks(range(len(draw_labels)), labels=draw_labels)

    return _create_plot_widget_with(qtbot, _plot)


def _create_plot_widget_with(qtbot: QtBot, plot: Callable[..., None]) -> PlotWidget:
    plotter = MagicMock()
    plotter.dimensionality = 1
    plotter.requires_observations = False
    plotter.plot.side_effect = plot
    plot_widget = PlotWidget("Any", plotter)
    qtbot.addWidget(plot_widget)
    _redraw(plot_widget)
    return plot_widget


def _redraw(plot_widget: PlotWidget) -> None:
    plot_widget.update_plot(
        PlotContext(
            PlotConfig(),
            ensembles=[],
            ensembles_color_indexes=[],
            key="key",
            layer=None,
        ),
        {},
        pd.DataFrame(),
        {},
        None,
    )


def _click_tick_labels(plot_widget: PlotWidget, axis: str) -> None:
    axes = plot_widget._figure.axes[0]
    tick_labels = (axes.xaxis if axis == "x" else axes.yaxis).get_ticklabels()
    renderer = plot_widget._canvas.get_renderer()
    x_pos, y_pos = (
        Bbox.union(
            [
                label.get_window_extent(renderer)
                for label in tick_labels
                if label.get_text()
            ]
        )
        .get_points()
        .mean(axis=0)
    )
    plot_widget._on_canvas_press(
        MouseEvent(
            "button_press_event",
            plot_widget._canvas,
            x_pos,
            y_pos,
            button=MouseButton.LEFT,
        )
    )


def _limits(plot_widget: PlotWidget, axis: str) -> tuple[float, float]:
    axes = plot_widget._figure.axes[0]
    return axes.get_xlim() if axis == "x" else axes.get_ylim()


def _accept_dialog_with(
    monkeypatch: pytest.MonkeyPatch, minimum: str, maximum: str
) -> list[AxisLimitsDialog]:
    opened: list[AxisLimitsDialog] = []

    def _exec(dialog: AxisLimitsDialog) -> int:
        opened.append(dialog)
        assert isinstance(dialog.minimum_edit, QLineEdit)
        assert isinstance(dialog.maximum_edit, QLineEdit)
        dialog.minimum_edit.setText(minimum)
        dialog.maximum_edit.setText(maximum)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(AxisLimitsDialog, "exec", _exec)
    return opened


@pytest.mark.parametrize("axis", ["x", "y"])
def test_that_clicking_tick_labels_opens_limits_dialog_and_sets_axis_limits(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, axis: str
):
    plot_widget = _create_plot_widget(qtbot, [1.0, 2.0, 3.0])
    opened = _accept_dialog_with(monkeypatch, "0.5", "1.5")

    _click_tick_labels(plot_widget, axis)

    assert len(opened) == 1
    assert opened[0].windowTitle() == f"Set {axis}-axis limits"
    assert _limits(plot_widget, axis) == (0.5, 1.5)


def test_that_user_axis_limits_are_kept_when_the_plot_is_redrawn(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
):
    plot_widget = _create_plot_widget(qtbot, [1.0, 2.0, 3.0])
    _accept_dialog_with(monkeypatch, "-10", "10")
    _click_tick_labels(plot_widget, "y")

    _redraw(plot_widget)

    assert _limits(plot_widget, "y") == (-10.0, 10.0)


def test_that_axis_limits_dialog_reset_restores_automatic_limits(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
):
    plot_widget = _create_plot_widget(qtbot, [1.0, 2.0, 3.0])
    automatic_limits = _limits(plot_widget, "y")
    _accept_dialog_with(monkeypatch, "-10", "10")
    _click_tick_labels(plot_widget, "y")

    def _reset(dialog: AxisLimitsDialog) -> int:
        dialog.reset_requested = True
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(AxisLimitsDialog, "exec", _reset)
    _click_tick_labels(plot_widget, "y")
    assert _limits(plot_widget, "y") == automatic_limits

    _redraw(plot_widget)
    assert _limits(plot_widget, "y") == automatic_limits


def test_that_user_axis_limits_keep_an_inverted_axis_inverted(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
):
    plot_widget = _create_plot_widget(qtbot, [1.0, 2.0, 3.0])
    plot_widget._figure.axes[0].invert_yaxis()
    plot_widget._canvas.draw()
    _accept_dialog_with(monkeypatch, "0", "5")

    _click_tick_labels(plot_widget, "y")

    assert _limits(plot_widget, "y") == (5.0, 0.0)


def test_that_cancelling_axis_limits_dialog_keeps_current_limits(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
):
    plot_widget = _create_plot_widget(qtbot, [1.0, 2.0, 3.0])
    limits_before = _limits(plot_widget, "x")
    monkeypatch.setattr(AxisLimitsDialog, "exec", lambda _: QDialog.DialogCode.Rejected)

    _click_tick_labels(plot_widget, "x")

    assert _limits(plot_widget, "x") == limits_before


@pytest.mark.parametrize(
    ("minimum", "maximum", "is_log", "expected_error"),
    [
        pytest.param("a", "1", False, "Limits must be numbers", id="not-a-number"),
        pytest.param(
            "2", "1", False, "Minimum must be less than maximum", id="reversed"
        ),
        pytest.param("1", "1", False, "Minimum must be less than maximum", id="equal"),
        pytest.param(
            "0",
            "1",
            True,
            "Limits must be positive on a logarithmic axis",
            id="non-positive-log",
        ),
        pytest.param("1e-3", "2.5", True, "", id="valid"),
    ],
)
def test_that_axis_limits_dialog_only_enables_ok_for_valid_limits(
    qtbot: QtBot, minimum: str, maximum: str, is_log: bool, expected_error: str
):
    dialog = AxisLimitsDialog("y", (1.0, 2.0), is_log=is_log)
    qtbot.addWidget(dialog)
    assert isinstance(dialog.minimum_edit, QLineEdit)
    assert isinstance(dialog.maximum_edit, QLineEdit)

    dialog.minimum_edit.setText(minimum)
    dialog.maximum_edit.setText(maximum)

    ok_button = dialog._buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert ok_button is not None
    assert ok_button.isEnabled() == (not expected_error)
    assert dialog._error_label.text() == expected_error


def test_that_axis_limits_dialog_shows_and_returns_dates_for_date_axes(
    qtbot: QtBot,
):
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2021, 6, 15, 12, 30, tzinfo=UTC)
    dialog = AxisLimitsDialog(
        "x",
        (float(mdates.date2num(start)), float(mdates.date2num(end))),
        is_date=True,
    )
    qtbot.addWidget(dialog)

    assert isinstance(dialog.minimum_edit, QDateTimeEdit)
    assert dialog.minimum_edit.text() == "2020-01-01 00:00:00"
    assert dialog.maximum_edit.text() == "2021-06-15 12:30:00"
    assert [mdates.num2date(value) for value in dialog.limits()] == [start, end]


def _select_categories(
    monkeypatch: pytest.MonkeyPatch, first: str, last: str
) -> list[list[str]]:
    offered_labels: list[list[str]] = []

    def _exec(dialog: AxisLimitsDialog) -> int:
        assert isinstance(dialog.minimum_edit, QComboBox)
        assert isinstance(dialog.maximum_edit, QComboBox)
        offered_labels.append(
            [
                dialog.minimum_edit.currentText(),
                dialog.maximum_edit.currentText(),
                *(
                    dialog.minimum_edit.itemText(i)
                    for i in range(dialog.minimum_edit.count())
                ),
            ]
        )
        dialog.minimum_edit.setCurrentText(first)
        dialog.maximum_edit.setCurrentText(last)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(AxisLimitsDialog, "exec", _exec)
    return offered_labels


def test_that_categorical_axis_limits_are_chosen_among_the_visible_tick_labels(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
):
    labels = ["0", "2", "3", "4", "6", "8"]
    plot_widget = _create_categorical_plot_widget(qtbot, [labels])
    offered_labels = _select_categories(monkeypatch, "2", "6")

    _click_tick_labels(plot_widget, "x")

    first, last, *choices = offered_labels[0]
    assert (first, last) == ("0", "8")
    assert choices == labels
    assert _limits(plot_widget, "x") == (0.5, 4.5)


def test_that_categorical_axis_limits_follow_the_chosen_labels_when_redrawn(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
):
    plot_widget = _create_categorical_plot_widget(
        qtbot, [["0", "2", "3", "4"], ["2", "3", "4", "5"]]
    )
    _select_categories(monkeypatch, "2", "3")
    _click_tick_labels(plot_widget, "x")
    assert _limits(plot_widget, "x") == (0.5, 2.5)

    _redraw(plot_widget)

    assert _limits(plot_widget, "x") == (-0.5, 1.5)


def test_that_axis_limits_dialog_rejects_a_first_category_after_the_last(
    qtbot: QtBot,
):
    dialog = AxisLimitsDialog("x", (-0.5, 1.5), categories=[(0.0, "a"), (1.0, "b")])
    qtbot.addWidget(dialog)
    assert isinstance(dialog.minimum_edit, QComboBox)
    assert isinstance(dialog.maximum_edit, QComboBox)

    dialog.minimum_edit.setCurrentIndex(1)
    dialog.maximum_edit.setCurrentIndex(0)

    assert dialog._error_label.text() == (
        "First category must not come after the last category"
    )
