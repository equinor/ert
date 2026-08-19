import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWIDGETSIZE_MAX, QLabel, QMainWindow
from pytestqt.qtbot import QtBot

from ert.gui.plotting.widgets.plot_side_panel import PlotSidePanel

MIN_WIDTH = 250


@pytest.fixture
def main_window(qtbot: QtBot) -> QMainWindow:
    window = QMainWindow()
    qtbot.addWidget(window)
    return window


def make_side_panel(
    main_window: QMainWindow, *, on_right: bool = False
) -> PlotSidePanel:
    side_panel = PlotSidePanel(
        "Plot controls", QLabel("content"), main_window, MIN_WIDTH, on_right=on_right
    )
    area = (
        Qt.DockWidgetArea.RightDockWidgetArea
        if on_right
        else Qt.DockWidgetArea.LeftDockWidgetArea
    )
    main_window.addDockWidget(area, side_panel)
    return side_panel


def test_that_collapsing_hides_content_but_keeps_toggle_button_visible(main_window):
    side_panel = make_side_panel(main_window)

    side_panel._toggle_button.setChecked(False)

    assert not side_panel._content.isVisible()
    assert not side_panel._title_label.isVisible()
    assert side_panel._toggle_button.isVisibleTo(side_panel.titleBarWidget())


def test_that_collapsing_fixes_width_to_toggle_button_width(main_window):
    side_panel = make_side_panel(main_window)

    side_panel._toggle_button.setChecked(False)

    expected = side_panel._toggle_button.sizeHint().width() + 8
    assert side_panel.minimumWidth() == expected
    assert side_panel.maximumWidth() == expected


def test_that_expanding_restores_minimum_width_and_unbounded_maximum(main_window):
    side_panel = make_side_panel(main_window)

    side_panel._toggle_button.setChecked(False)
    side_panel._toggle_button.setChecked(True)

    assert side_panel.minimumWidth() == MIN_WIDTH
    assert side_panel.maximumWidth() == QWIDGETSIZE_MAX


def test_that_expanding_restores_the_width_the_side_panel_had_before_collapsing(
    qtbot: QtBot, main_window: QMainWindow
) -> None:
    main_window.setCentralWidget(QLabel("plot area"))
    side_panel = make_side_panel(main_window)
    main_window.resize(1000, 600)
    with qtbot.waitExposed(main_window):
        main_window.show()
    main_window.resizeDocks([side_panel], [400], Qt.Orientation.Horizontal)
    qtbot.waitUntil(lambda: side_panel.width() > side_panel._min_expanded_width)
    width_before_collapse = side_panel.width()

    side_panel._toggle_button.setChecked(False)
    qtbot.waitUntil(lambda: not side_panel._content.isVisible())
    side_panel._toggle_button.setChecked(True)

    qtbot.waitUntil(lambda: side_panel.width() == width_before_collapse)


@pytest.mark.parametrize(
    ("on_right", "expected_object_name"),
    [(False, "left_plot_side_panel"), (True, "right_plot_side_panel")],
)
def test_that_side_panel_object_name_and_icons_are_side_specific(
    main_window, on_right, expected_object_name
):
    side_panel = make_side_panel(main_window, on_right=on_right)

    assert side_panel.objectName() == expected_object_name
    assert not side_panel._collapse_icon.isNull()


def test_that_toggle_button_is_placed_towards_the_plot_area(main_window):
    left = make_side_panel(main_window, on_right=False)
    right = make_side_panel(main_window, on_right=True)

    left_layout = left.titleBarWidget().layout()
    right_layout = right.titleBarWidget().layout()

    assert left_layout.itemAt(left_layout.count() - 1).widget() is left._toggle_button
    assert right_layout.itemAt(0).widget() is right._toggle_button


def test_that_tooltip_describes_the_action_the_button_will_perform(main_window):
    side_panel = make_side_panel(main_window)

    assert side_panel._toggle_button.toolTip() == "Collapse Plot controls panel"

    side_panel._toggle_button.setChecked(False)

    assert side_panel._toggle_button.toolTip() == "Open Plot controls panel"
