from PyQt6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QLabel,
    QVBoxLayout,
)

from ert.gui.plotting.utils.qt_creator import (
    create_group_box,
    create_group_layout,
    create_side_panel,
)


def test_that_create_group_layout_creates_layout_with_widgets(qtbot):
    mock_widget1 = QCheckBox("Widget 1")
    mock_widget2 = QCheckBox("Widget 2")
    widgets = [mock_widget1, mock_widget2]

    for widget in widgets:
        qtbot.addWidget(widget)

    layout = create_group_layout(widgets)

    assert isinstance(layout, QVBoxLayout)

    assert layout.count() == len(widgets)

    for i in range(layout.count()):
        assert layout.itemAt(i).widget() == widgets[i]


def test_that_create_group_box_creates_group_box_with_layout(qtbot):
    mock_layout = QVBoxLayout()

    title = "test group box"
    group_box = create_group_box(title, mock_layout)
    qtbot.addWidget(group_box)

    assert isinstance(group_box, QGroupBox)
    assert group_box.title() == title
    assert group_box.layout() == mock_layout


def test_that_create_side_panel_creates_panel_with_title_and_widget(qtbot):
    checkbox_widget = QCheckBox("checkbox widget")
    qtbot.addWidget(checkbox_widget)

    title = "test panel"
    panel = create_side_panel(title, checkbox_widget)
    qtbot.addWidget(panel)

    assert panel.layout().count() == 2

    title_label = panel.layout().itemAt(0).widget()
    assert title_label.text() == title
    assert isinstance(title_label, QLabel)

    content_widget = panel.layout().itemAt(1).widget()
    assert content_widget == checkbox_widget
    assert isinstance(content_widget, QCheckBox)
