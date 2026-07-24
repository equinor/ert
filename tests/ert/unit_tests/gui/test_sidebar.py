from collections.abc import Iterator
from typing import cast

import pytest
from PyQt6.QtCore import QCoreApplication, QEvent, Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QToolButton,
    QWidget,
)
from pytestqt.qtbot import QtBot

from ert.gui.main import apply_application_theme
from ert.gui.sidebar import (
    COLLAPSE_COLOR_TOKEN,
    COLLAPSE_SIDEBAR,
    COLLAPSED_WIDTH,
    CREATE_PLOT,
    EXPAND_SIDEBAR,
    EXPANDED_WIDTH,
    EXPERIMENT_STATUS,
    HEADER_OBJECT_NAME,
    MANAGE_EXPERIMENTS,
    NAV_COLOR_CHECKED_TOKEN,
    NAV_COLOR_DEFAULT_TOKEN,
    NAV_COLOR_DISABLED_TOKEN,
    NAV_GROUP_OBJECT_NAME,
    NAV_ITEM_OBJECT_NAME,
    NAVIGATION_ENTRIES,
    SIDEBAR_TITLE,
    START_EXPERIMENT,
    TITLE_OBJECT_NAME,
    Sidebar,
    _detect_scheme,
    object_name_for_entry,
)
from ert.gui.theming.theme import ColorScheme, resolve_color

ALL_ENTRIES = [START_EXPERIMENT, CREATE_PLOT, MANAGE_EXPERIMENTS, EXPERIMENT_STATUS]


@pytest.fixture
def sidebar(qtbot: QtBot) -> Sidebar:
    bar = Sidebar()
    qtbot.addWidget(bar)
    return bar


def _collapse_button(sidebar: Sidebar) -> QToolButton:
    button = sidebar.findChild(QToolButton, "button_collapse_sidebar")
    assert isinstance(button, QToolButton)
    return button


def _button(sidebar: Sidebar, name: str) -> QToolButton:
    button = sidebar.findChild(QToolButton, object_name_for_entry(name))
    assert isinstance(button, QToolButton)
    return button


def _icon_tint_hex(button: QToolButton) -> str:
    """Return the ``#rrggbb`` of the button icon's dominant fully-opaque pixel.

    Sidebar icons are solid-filled SVGs, so the most common opaque colour is the
    tint applied via ``load_icon(color=...)``.
    """
    image = button.icon().pixmap(24, 24).toImage()
    counts: dict[str, int] = {}
    for y in range(image.height()):
        for x in range(image.width()):
            pixel = image.pixelColor(x, y)
            if pixel.alpha() == 255:
                key = pixel.name()
                counts[key] = counts.get(key, 0) + 1
    assert counts, "icon has no fully opaque pixels to sample"
    return max(counts, key=lambda color: counts[color])


def test_that_triggering_an_action_emits_page_requested_with_its_name(
    sidebar: Sidebar,
    qtbot: QtBot,
):
    for name in ALL_ENTRIES:
        with qtbot.wait_signal(sidebar.page_requested) as blocker:
            sidebar.action_for(name).trigger()
        assert blocker.args == [name]


def test_that_set_current_checks_only_the_named_action(sidebar: Sidebar):
    sidebar.set_current(EXPERIMENT_STATUS)

    assert sidebar.action_for(EXPERIMENT_STATUS).isChecked()
    for name in ALL_ENTRIES:
        if name != EXPERIMENT_STATUS:
            assert not sidebar.action_for(name).isChecked()


def test_that_set_current_does_not_emit_page_requested(sidebar: Sidebar):
    emitted: list[str] = []
    sidebar.page_requested.connect(emitted.append)

    sidebar.set_current(MANAGE_EXPERIMENTS)

    assert emitted == []


def test_that_set_status_enabled_toggles_the_experiment_status_action(sidebar: Sidebar):
    sidebar.set_status_enabled(False)
    assert not sidebar.action_for(EXPERIMENT_STATUS).isEnabled()

    sidebar.set_status_enabled(True)
    assert sidebar.action_for(EXPERIMENT_STATUS).isEnabled()


@pytest.mark.parametrize(
    ("enabled", "shape"),
    [
        pytest.param(
            False,
            Qt.CursorShape.ForbiddenCursor,
            id="disabled_shows_forbidden_cursor",
        ),
        pytest.param(
            True,
            Qt.CursorShape.PointingHandCursor,
            id="enabled_shows_pointing_hand_cursor",
        ),
    ],
)
def test_that_nav_button_cursor_reflects_its_enabled_state(
    sidebar: Sidebar, enabled, shape
):
    sidebar.set_status_enabled(False)
    sidebar.set_status_enabled(enabled)

    button = _button(sidebar, EXPERIMENT_STATUS)
    assert button.isEnabled() is enabled
    assert button.cursor().shape() == shape


def test_that_collapse_button_shows_pointing_hand_cursor(sidebar: Sidebar):
    assert sidebar._collapse_button is not None
    assert (
        sidebar._collapse_button.cursor().shape() == Qt.CursorShape.PointingHandCursor
    )


def _send_container_event(sidebar: Sidebar, name: str, event_type: QEvent.Type):
    container = _button(sidebar, name).parent()
    assert isinstance(container, QFrame)
    QCoreApplication.sendEvent(container, QEvent(event_type))
    return _button(sidebar, name)


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        pytest.param(
            [QEvent.Type.Enter],
            "true",
            id="entering_marks_inner_button_hovered",
        ),
        pytest.param(
            [QEvent.Type.Enter, QEvent.Type.Leave],
            "false",
            id="leaving_clears_inner_button_hovered",
        ),
    ],
)
def test_that_container_hover_events_drive_the_inner_button_hovered(
    sidebar: Sidebar, events, expected
):
    button = _button(sidebar, START_EXPERIMENT)
    for event_type in events:
        button = _send_container_event(sidebar, START_EXPERIMENT, event_type)

    assert button.property("hovered") == expected


def test_that_hovering_a_disabled_container_does_not_mark_the_button_hovered(
    sidebar: Sidebar,
):
    sidebar.set_status_enabled(False)
    button = _send_container_event(sidebar, EXPERIMENT_STATUS, QEvent.Type.Enter)

    assert button.property("hovered") == "false"


@pytest.mark.parametrize(
    ("enabled", "nav_disabled"),
    [
        pytest.param(False, "true", id="disabling_sets_nav_disabled_true"),
        pytest.param(True, "false", id="reenabling_sets_nav_disabled_false"),
    ],
)
def test_that_nav_item_container_nav_disabled_reflects_enabled_state(
    sidebar: Sidebar, enabled, nav_disabled
):
    sidebar.set_status_enabled(False)
    sidebar.set_status_enabled(enabled)

    container = _button(sidebar, EXPERIMENT_STATUS).parent()
    assert container.property("nav_disabled") == nav_disabled


def test_that_right_clicking_the_create_plot_button_emits_external_plot_requested(
    sidebar: Sidebar,
    qtbot: QtBot,
):
    with qtbot.wait_signal(sidebar.external_plot_requested, timeout=1000):
        qtbot.mouseClick(_button(sidebar, CREATE_PLOT), Qt.MouseButton.RightButton)


def test_that_left_click_on_create_plot_button_does_not_emit_external_plot_request(
    sidebar: Sidebar,
    qtbot: QtBot,
):
    emitted: list[None] = []
    sidebar.external_plot_requested.connect(lambda: emitted.append(None))

    qtbot.mouseClick(_button(sidebar, CREATE_PLOT), Qt.MouseButton.LeftButton)

    assert emitted == []


def test_that_action_for_returns_the_action_registered_for_a_name(sidebar: Sidebar):
    action = sidebar.action_for(CREATE_PLOT)

    assert isinstance(action, QAction)
    assert action.isCheckable()
    assert action.toolTip() == CREATE_PLOT


def test_that_object_name_for_entry_replaces_spaces_with_underscores():
    assert object_name_for_entry(START_EXPERIMENT) == "button_Start_experiment"
    assert object_name_for_entry(EXPERIMENT_STATUS) == "button_Experiment_status"


def test_that_configure_toolbar_makes_the_rail_vertical_and_immovable(sidebar: Sidebar):
    assert sidebar.objectName() == "sidebar"
    assert sidebar.orientation() == Qt.Orientation.Vertical
    assert sidebar.isMovable() is False
    assert sidebar.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonTextBesideIcon


def test_that_navigation_entries_table_matches_the_created_buttons(sidebar: Sidebar):
    entry_names = [name for name, _icon_file in NAVIGATION_ENTRIES]

    assert entry_names == ALL_ENTRIES
    for name in entry_names:
        assert _button(sidebar, name) is not None


def test_that_each_nav_button_is_wrapped_in_its_own_outer_container(sidebar: Sidebar):
    nav_group = sidebar.findChild(QFrame, NAV_GROUP_OBJECT_NAME)
    assert nav_group is not None

    for name in ALL_ENTRIES:
        button = _button(sidebar, name)
        container = button.parent()
        assert isinstance(container, QFrame)
        assert container.objectName() == NAV_ITEM_OBJECT_NAME
        assert container.parent() is nav_group


def test_that_nav_item_container_keeps_a_4px_gap_around_the_button(sidebar: Sidebar):
    for name in ALL_ENTRIES:
        container = _button(sidebar, name).parent()
        assert isinstance(container, QFrame)
        margins = container.layout().contentsMargins()
        assert (
            margins.left(),
            margins.top(),
            margins.right(),
            margins.bottom(),
        ) == (4, 4, 4, 4)


def test_that_selecting_a_nav_item_sets_checked_property_only_on_its_container(
    sidebar: Sidebar,
):
    sidebar.set_current(CREATE_PLOT)

    for name in ALL_ENTRIES:
        container = _button(sidebar, name).parent()
        assert isinstance(container, QFrame)
        expected = "true" if name == CREATE_PLOT else "false"
        assert container.property("checked") == expected


def test_that_switching_nav_selection_clears_checked_property_on_the_old_container(
    sidebar: Sidebar,
):
    sidebar.set_current(CREATE_PLOT)
    sidebar.set_current(MANAGE_EXPERIMENTS)

    assert _button(sidebar, CREATE_PLOT).parent().property("checked") == "false"
    assert _button(sidebar, MANAGE_EXPERIMENTS).parent().property("checked") == "true"


def test_that_create_plot_button_has_the_right_click_tooltip(sidebar: Sidebar):
    assert (
        _button(sidebar, CREATE_PLOT).toolTip() == "Right click to open external window"
    )


def test_that_collapse_button_is_inside_the_first_toolbar_widget(sidebar: Sidebar):
    first_action = sidebar.actions()[0]
    header = sidebar.widgetForAction(first_action)

    assert header is _collapse_button(sidebar).parent()


def test_that_sidebar_starts_expanded_with_text_beside_icons(sidebar: Sidebar):
    assert sidebar.collapsed is False
    assert sidebar.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonTextBesideIcon


def test_that_clicking_collapse_button_switches_to_icon_only_mode(
    sidebar: Sidebar,
    qtbot: QtBot,
):
    qtbot.mouseClick(_collapse_button(sidebar), Qt.MouseButton.LeftButton)

    assert sidebar.collapsed is True
    assert sidebar.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonIconOnly


def test_that_clicking_collapse_button_twice_restores_text_beside_icons(
    sidebar: Sidebar,
    qtbot: QtBot,
):
    qtbot.mouseClick(_collapse_button(sidebar), Qt.MouseButton.LeftButton)
    qtbot.mouseClick(_collapse_button(sidebar), Qt.MouseButton.LeftButton)

    assert sidebar.collapsed is False
    assert sidebar.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonTextBesideIcon


def test_that_collapse_button_is_not_part_of_the_navigation_action_group(
    sidebar: Sidebar,
):
    collapse_action = _collapse_button(sidebar).defaultAction()

    assert collapse_action.actionGroup() is None
    assert collapse_action.isCheckable() is False


def test_that_collapse_button_stays_icon_only_after_expanding_again(sidebar: Sidebar):
    button = _collapse_button(sidebar)

    sidebar.set_collapsed(True)
    sidebar.set_collapsed(False)

    assert button.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonIconOnly


def test_that_collapse_button_tooltip_reflects_next_toggle_action(sidebar: Sidebar):
    button = _collapse_button(sidebar)
    assert button.toolTip() == COLLAPSE_SIDEBAR

    sidebar.set_collapsed(True)
    assert button.toolTip() == EXPAND_SIDEBAR

    sidebar.set_collapsed(False)
    assert button.toolTip() == COLLAPSE_SIDEBAR


def test_that_collapse_icon_rotates_180_degrees_when_toggled(sidebar: Sidebar):
    button = _collapse_button(sidebar)
    expanded = button.icon().pixmap(24, 24).toImage()

    sidebar.set_collapsed(True)
    collapsed = button.icon().pixmap(24, 24).toImage()

    assert collapsed != expanded
    assert collapsed == expanded.mirrored(True, True)


def _title_label(sidebar: Sidebar) -> QLabel:
    label = sidebar.findChild(QLabel, TITLE_OBJECT_NAME)
    assert isinstance(label, QLabel)
    return label


def test_that_sidebar_shows_a_title_label_with_the_application_name(sidebar: Sidebar):
    assert _title_label(sidebar).text() == SIDEBAR_TITLE


def test_that_title_label_appears_before_the_navigation_group(sidebar: Sidebar):
    title = _title_label(sidebar)

    widgets = [sidebar.widgetForAction(action) for action in sidebar.actions()]
    header = title.parent()
    title_index = widgets.index(header)
    nav_container = sidebar.findChild(QFrame, NAV_GROUP_OBJECT_NAME)
    assert nav_container is not None
    nav_index = widgets.index(nav_container)

    assert title_index < nav_index


def test_that_title_shares_the_header_row_with_the_collapse_button(sidebar: Sidebar):
    assert _title_label(sidebar).parent() is _collapse_button(sidebar).parent()


def test_that_title_is_hidden_when_collapsed_and_shown_when_expanded(sidebar: Sidebar):
    sidebar.show()

    title = _title_label(sidebar)
    assert title.isVisible()

    sidebar.set_collapsed(True)
    assert not title.isVisible()

    sidebar.set_collapsed(False)
    assert title.isVisible()


def test_that_sidebar_uses_compact_expanded_and_icon_rail_collapsed_widths(
    sidebar: Sidebar,
):
    assert sidebar.width() == EXPANDED_WIDTH

    sidebar.set_collapsed(True)
    assert sidebar.width() == COLLAPSED_WIDTH

    sidebar.set_collapsed(False)
    assert sidebar.width() == EXPANDED_WIDTH


def test_that_nav_buttons_are_icon_only_when_collapsed(sidebar: Sidebar):
    sidebar.set_collapsed(True)

    for name in ALL_ENTRIES:
        assert (
            _button(sidebar, name).toolButtonStyle()
            == Qt.ToolButtonStyle.ToolButtonIconOnly
        )


def test_that_header_and_nav_layouts_keep_compact_spacing(sidebar: Sidebar):
    header = sidebar.findChild(QWidget, HEADER_OBJECT_NAME)
    nav_group = sidebar.findChild(QFrame, NAV_GROUP_OBJECT_NAME)
    assert header is not None
    assert nav_group is not None

    header_layout = header.layout()
    nav_layout = nav_group.layout()
    assert header_layout is not None
    assert nav_layout is not None

    assert header_layout.contentsMargins().left() == 8
    assert nav_layout.contentsMargins().left() == 8
    assert nav_layout.spacing() == 30

    sidebar.set_collapsed(True)
    assert header_layout.spacing() == 0
    assert nav_layout.spacing() == 30


def test_that_header_and_nav_margins_are_unchanged_by_collapsing(sidebar: Sidebar):
    header_layout = sidebar.findChild(QWidget, HEADER_OBJECT_NAME).layout()
    nav_layout = sidebar.findChild(QFrame, NAV_GROUP_OBJECT_NAME).layout()
    assert header_layout is not None
    assert nav_layout is not None

    def _margins(layout):
        m = layout.contentsMargins()
        return (m.left(), m.top(), m.right(), m.bottom())

    expanded = (_margins(header_layout), _margins(nav_layout))

    sidebar.set_collapsed(True)

    assert (_margins(header_layout), _margins(nav_layout)) == expanded


@pytest.mark.parametrize(
    ("prepare", "target", "token"),
    [
        pytest.param(
            lambda _s: None,
            START_EXPERIMENT,
            NAV_COLOR_DEFAULT_TOKEN,
            id="default_state_uses_accent_strong",
        ),
        pytest.param(
            lambda s: s.set_current(CREATE_PLOT),
            CREATE_PLOT,
            NAV_COLOR_CHECKED_TOKEN,
            id="selected_uses_checked_text_color",
        ),
        pytest.param(
            lambda s: s.set_status_enabled(False),
            EXPERIMENT_STATUS,
            NAV_COLOR_DISABLED_TOKEN,
            id="disabled_uses_disabled_text_color",
        ),
        pytest.param(
            lambda s: (s.set_current(CREATE_PLOT), s.set_current(MANAGE_EXPERIMENTS)),
            CREATE_PLOT,
            NAV_COLOR_DEFAULT_TOKEN,
            id="deselected_restores_default_color",
        ),
    ],
)
def test_that_nav_icon_tint_matches_its_buttons_state(
    sidebar: Sidebar, prepare, target, token
):
    sidebar.retint_all(ColorScheme.LIGHT)

    prepare(sidebar)

    expected = resolve_color(ColorScheme.LIGHT, token)
    assert _icon_tint_hex(_button(sidebar, target)) == expected


def test_that_retint_all_recolors_icons_for_the_new_color_scheme(sidebar: Sidebar):
    sidebar.retint_all(ColorScheme.DARK)

    expected = resolve_color(ColorScheme.DARK, NAV_COLOR_DEFAULT_TOKEN)
    assert _icon_tint_hex(_button(sidebar, START_EXPERIMENT)) == expected


def test_that_collapse_icon_uses_the_neutral_strong_text_color(sidebar: Sidebar):
    sidebar.retint_all(ColorScheme.LIGHT)

    expected = resolve_color(ColorScheme.LIGHT, COLLAPSE_COLOR_TOKEN)
    assert _icon_tint_hex(_collapse_button(sidebar)) == expected


def test_that_detect_scheme_falls_back_to_light_when_detection_raises(monkeypatch):
    def _raise() -> ColorScheme:
        raise RuntimeError("no QApplication")

    monkeypatch.setattr("ert.gui.sidebar.detect_system_color_scheme", _raise)

    assert _detect_scheme() == ColorScheme.LIGHT


def test_that_hover_event_on_an_unknown_container_is_ignored(
    sidebar: Sidebar,
    qtbot: QtBot,
):
    stray = QFrame()
    qtbot.addWidget(stray)

    handled = sidebar.eventFilter(stray, QEvent(QEvent.Type.Enter))

    assert handled is False


def test_that_retinting_without_a_collapse_action_still_tints_nav_icons(
    sidebar: Sidebar,
):
    sidebar._collapse_action = None

    sidebar.retint_all(ColorScheme.DARK)

    expected = resolve_color(ColorScheme.DARK, NAV_COLOR_DEFAULT_TOKEN)
    assert _icon_tint_hex(_button(sidebar, START_EXPERIMENT)) == expected


def test_that_button_for_returns_the_button_for_a_known_name(sidebar: Sidebar):
    assert sidebar.button_for(START_EXPERIMENT) is _button(sidebar, START_EXPERIMENT)


def test_that_button_for_returns_none_for_an_unknown_name(sidebar: Sidebar):
    assert sidebar.button_for("not a nav entry") is None


def _status_menu(sidebar: Sidebar) -> QMenu:
    button = sidebar.findChild(QToolButton, object_name_for_entry(EXPERIMENT_STATUS))
    assert isinstance(button, QToolButton)
    menu = button.menu()
    assert isinstance(menu, QMenu)
    return menu


def test_that_a_second_status_entry_builds_a_dropdown_with_both_entries(
    sidebar: Sidebar,
):
    sidebar.add_status_entry("first")
    sidebar.add_status_entry("second")

    labels = [action.text() for action in _status_menu(sidebar).actions()]
    assert labels == ["second", "first"]


def test_that_a_third_status_entry_is_prepended_to_the_dropdown(sidebar: Sidebar):
    for name in ("first", "second", "third"):
        sidebar.add_status_entry(name)

    labels = [action.text() for action in _status_menu(sidebar).actions()]
    assert labels == ["third", "second", "first"]


def test_that_triggering_a_status_menu_entry_emits_status_entry_selected(
    sidebar: Sidebar,
    qtbot: QtBot,
):
    sidebar.add_status_entry("first")
    sidebar.add_status_entry("second")

    with qtbot.wait_signal(sidebar.status_entry_selected) as blocker:
        _status_menu(sidebar).actions()[0].trigger()

    assert blocker.args == ["second"]


def test_that_the_most_recently_added_status_entry_is_the_only_bold_one(
    sidebar: Sidebar,
):
    sidebar.add_status_entry("first")
    sidebar.add_status_entry("second")

    bold = {a.text(): a.font().bold() for a in _status_menu(sidebar).actions()}
    assert bold == {"second": True, "first": False}


def test_that_triggering_a_status_entry_makes_only_that_entry_bold(
    sidebar: Sidebar,
):
    sidebar.add_status_entry("first")
    sidebar.add_status_entry("second")

    next(a for a in _status_menu(sidebar).actions() if a.text() == "first").trigger()

    bold = {a.text(): a.font().bold() for a in _status_menu(sidebar).actions()}
    assert bold == {"first": True, "second": False}


def test_that_hiding_the_status_menu_clears_the_status_button_hover_state(
    sidebar: Sidebar,
):
    sidebar.add_status_entry("first")
    sidebar.add_status_entry("second")

    button = _send_container_event(sidebar, EXPERIMENT_STATUS, QEvent.Type.Enter)
    assert button.property("hovered") == "true"

    _status_menu(sidebar).aboutToHide.emit()

    assert button.property("hovered") == "false"


def test_that_status_menu_helpers_build_no_menu_without_a_status_button(
    sidebar: Sidebar,
):
    original_status_button = sidebar._status_button
    assert original_status_button is not None
    sidebar._status_button = None

    sidebar._create_status_menu()
    sidebar._add_status_menu_item("orphan")
    sidebar._on_status_menu_about_to_hide()

    assert sidebar._status_button is None
    assert original_status_button.menu() is None


def test_that_adding_a_status_menu_item_before_the_menu_exists_is_a_noop(
    sidebar: Sidebar,
):
    assert sidebar._status_button is not None
    assert sidebar._status_button.menu() is None

    sidebar._add_status_menu_item("orphan")

    assert sidebar._status_button.menu() is None


@pytest.fixture
def themed_sidebar(qtbot: QtBot) -> Iterator[Sidebar]:
    """A sidebar laid out inside a host widget with the application stylesheet.

    Geometry assertions are only meaningful with the real QSS applied, because
    the stylesheet sets the fonts the labels are measured with, and only a
    non-top-level sidebar is actually held at ``EXPANDED_WIDTH``.
    """
    app = cast(QApplication, QApplication.instance())
    previous_stylesheet = app.styleSheet()
    apply_application_theme(app)

    host = QWidget()
    layout = QHBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    bar = Sidebar(host)
    layout.addWidget(bar)
    layout.addWidget(QLabel("content", host))
    host.resize(1200, 700)
    qtbot.addWidget(host)
    host.show()
    qtbot.waitExposed(host)
    try:
        yield bar
    finally:
        app.setStyleSheet(previous_stylesheet)


def test_that_expanded_rail_shows_every_navigation_label_without_eliding_it(
    themed_sidebar: Sidebar,
):
    assert themed_sidebar.width() == EXPANDED_WIDTH

    for name in ALL_ENTRIES:
        button = themed_sidebar.button_for(name)
        assert button is not None
        assert button.sizeHint().width() <= button.width(), (
            f"{name!r} needs {button.sizeHint().width()}px but only has "
            f"{button.width()}px, so Qt elides it"
        )


def test_that_header_keeps_its_height_when_the_sidebar_collapses(
    themed_sidebar: Sidebar, qtbot: QtBot
):
    header = themed_sidebar.findChild(QWidget, HEADER_OBJECT_NAME)
    assert header is not None
    expanded_height = header.height()

    themed_sidebar.set_collapsed(True)
    qtbot.wait(1)

    assert header.height() == expanded_height, (
        "hiding the title on collapse shrank the header, "
        "which shifts the navigation entries vertically"
    )
