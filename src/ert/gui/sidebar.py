from __future__ import annotations

from typing import override

from PyQt6.QtCore import QEvent, QObject, QSize, Qt
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtGui import QAction, QActionGroup, QIcon, QMouseEvent
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QSizePolicy,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .icon_utils import load_icon
from .theming.manager import detect_system_color_scheme
from .theming.theme import ColorScheme, resolve_color

# Compact expanded rail matching the destination-item mockup; collapsed shows
# icons only. The expanded width fits the widest label ("Manage experiments",
# 215px with the sidebar font) beside a 24px icon, plus the nav container and
# toolbar insets and a small margin so slight font metric differences between
# platforms cannot clip it; at 220px it was clipped to "Manage ...riments".
EXPANDED_WIDTH = 256
COLLAPSED_WIDTH = 100

SIDEBAR_TITLE = "ERT"

START_EXPERIMENT = "Start experiment"
CREATE_PLOT = "Create plot"
MANAGE_EXPERIMENTS = "Manage experiments"
EXPERIMENT_STATUS = "Experiment status"
COLLAPSE_SIDEBAR = "Collapse sidebar"
EXPAND_SIDEBAR = "Expand sidebar"
COLLAPSE_ICON = "collapse_sidebar.svg"

# Object name of the QFrame that wraps the navigation buttons so QSS can target
# the action group without styling the collapse control.
NAV_GROUP_OBJECT_NAME = "sidebar_nav_group"

# Object name of the QFrame that wraps each individual navigation button so QSS
# can style the outer container separately from the inner action button.
NAV_ITEM_OBJECT_NAME = "sidebar_nav_item"

# Object name of the sidebar title label so QSS can set its font size.
TITLE_OBJECT_NAME = "sidebar_title"

# Object name of the header row that holds the collapse button and the title.
HEADER_OBJECT_NAME = "sidebar_header"

# Object name of the status pop-up menu so QSS can set its item font size.
STATUS_MENU_OBJECT_NAME = "sidebar_status_menu"

# Layout insets: kept identical across collapse states so labels and icons share
# the same padding; only the width, spacing, and button style change on collapse.
_HEADER_MARGINS = (8, 8, 8, 4)
_HEADER_SPACING_EXPANDED = 4
_HEADER_SPACING_COLLAPSED = 0
_NAV_MARGINS = (8, 4, 8, 8)
_NAV_SPACING = 30

NAVIGATION_ENTRIES: tuple[tuple[str, str], ...] = (
    (START_EXPERIMENT, "library_add.svg"),
    (CREATE_PLOT, "timeline.svg"),
    (MANAGE_EXPERIMENTS, "build_wrench.svg"),
    (EXPERIMENT_STATUS, "in_progress.svg"),
)

# EDS text-colour tokens each icon is tinted with so it matches the QSS text
# colour of its button in every interaction state (see theme.qss.in).
NAV_COLOR_DEFAULT_TOKEN = "text-accent-strong"
NAV_COLOR_CHECKED_TOKEN = "bg-accent-fill-emphasis-active"
NAV_COLOR_DISABLED_TOKEN = "text-disabled"
COLLAPSE_COLOR_TOKEN = "text-neutral-strong"


def _detect_scheme() -> ColorScheme:
    """Best-effort current colour scheme, defaulting to light when headless."""
    try:
        return detect_system_color_scheme()
    except RuntimeError:
        return ColorScheme.LIGHT


def object_name_for_entry(name: str) -> str:
    """Reproduce the button object name the previous sidebar exposed.

    The new toolbar-based sidebar keeps these ids (e.g. "Start experiment" ->
    "button_Start_experiment") so existing GUI tests and QSS that look buttons
    up by object name keep working during the migration.
    """
    return f"button_{name.replace(' ', '_')}"


class Sidebar(QToolBar):
    page_requested = Signal(str)
    external_plot_requested = Signal()
    status_entry_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._action_group = QActionGroup(self)
        self._action_group.setExclusive(True)
        self._actions: dict[str, QAction] = {}
        self._nav_buttons: dict[str, QToolButton] = {}
        self._nav_icon_files: dict[str, str] = {}
        self._color_scheme: ColorScheme = _detect_scheme()
        self._nav_containers: dict[str, QFrame] = {}
        self._nav_group: QFrame | None = None
        self._header: QWidget | None = None
        self._header_layout: QHBoxLayout | None = None
        self._nav_layout: QVBoxLayout | None = None
        self._title_label: QLabel | None = None
        self._collapse_button: QToolButton | None = None
        self._plot_button: QToolButton | None = None
        self._status_button: QToolButton | None = None
        self._status_entries: list[str] = []
        self._collapse_action: QAction | None = None
        self._collapsed = False

        self._configure_toolbar()
        self._build_header()
        self._build_navigation_entries()
        self._enable_external_plot_trigger()
        self._capture_status_button()
        self._apply_layout_for_collapse_state()
        self._tint_all_nav_icons()

    def _configure_toolbar(self) -> None:
        self.setObjectName("sidebar")
        self.setOrientation(Qt.Orientation.Vertical)
        self.setMovable(False)
        self.setFixedWidth(EXPANDED_WIDTH)
        self.setIconSize(QSize(24, 24))
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

    def _build_header(self) -> None:
        """Build the header row with the collapse button next to the title."""
        header = QWidget(self)
        header.setObjectName(HEADER_OBJECT_NAME)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(*_HEADER_MARGINS)
        layout.setSpacing(_HEADER_SPACING_EXPANDED)

        action = QAction(
            load_icon(
                COLLAPSE_ICON,
                color=resolve_color(self._color_scheme, COLLAPSE_COLOR_TOKEN),
            ),
            COLLAPSE_SIDEBAR,
            self,
        )
        action.setToolTip(COLLAPSE_SIDEBAR)
        action.triggered.connect(lambda _=False: self._toggle_collapsed())
        self._collapse_action = action

        button = QToolButton(header)
        button.setObjectName("button_collapse_sidebar")
        button.setDefaultAction(action)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._collapse_button = button

        label = QLabel(SIDEBAR_TITLE, header)
        label.setObjectName(TITLE_OBJECT_NAME)
        self._title_label = label

        layout.addWidget(button)
        layout.addWidget(label)
        layout.addStretch()

        self._header = header
        self._header_layout = layout
        self._reserve_header_height()
        self.addWidget(header)

    def _reserve_header_height(self) -> None:
        """Freeze the header height at its expanded size.

        Collapsing hides the title, which would otherwise shrink the header row
        and slide the whole navigation group upwards. Reserving the height the
        title needs keeps the nav entries anchored in both states. It is derived
        from the label's own size hint so it tracks the QSS font size instead of
        duplicating it as a second constant.
        """
        if self._header is None or self._title_label is None:
            return
        _, top, _, bottom = _HEADER_MARGINS
        collapse_height = (
            self._collapse_button.sizeHint().height()
            if self._collapse_button is not None
            else 0
        )
        content_height = max(self._title_label.sizeHint().height(), collapse_height)
        self._header.setFixedHeight(content_height + top + bottom)

    def _build_navigation_entries(self) -> None:
        container = QFrame(self)
        container.setObjectName(NAV_GROUP_OBJECT_NAME)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(*_NAV_MARGINS)
        layout.setSpacing(_NAV_SPACING)
        for name, icon_file in NAVIGATION_ENTRIES:
            self._nav_icon_files[name] = icon_file
            button = self._create_nav_button(name, load_icon(icon_file))
            item = self._wrap_nav_button(button)
            self._nav_containers[name] = item
            action = self._actions[name]
            action.toggled.connect(
                lambda checked, c=item: self._set_container_checked(c, checked)
            )
            action.changed.connect(
                lambda c=item, a=action: self._sync_container_disabled(c, a.isEnabled())
            )
            self._sync_container_disabled(item, action.isEnabled())
            layout.addWidget(item)
        layout.addStretch()
        self._nav_group = container
        self._nav_layout = layout
        self.addWidget(container)

    def _wrap_nav_button(self, button: QToolButton) -> QFrame:
        """Wrap a navigation button in its own styleable outer container.

        The container is a ``QFrame`` (not a bare ``QWidget``) so QSS can paint
        its ``background-color`` directly, matching how the nav group frame is
        styled.
        """
        container = QFrame(self)
        container.setObjectName(NAV_ITEM_OBJECT_NAME)
        container.setProperty("checked", "false")
        container.installEventFilter(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)
        layout.addWidget(button)
        return container

    def _set_container_checked(self, container: QFrame, checked: bool) -> None:
        """Mirror a nav button's selected state onto its outer container.

        Qt cannot style a parent from a child's ``:checked`` pseudo-state, so
        the container carries a ``checked`` property that QSS targets to paint
        the active outer-container background.
        """
        container.setProperty("checked", "true" if checked else "false")
        style = container.style()
        if style is not None:
            style.unpolish(container)
            style.polish(container)

    def _sync_container_disabled(self, container: QFrame, enabled: bool) -> None:
        """Mirror a nav button's disabled state onto its outer container.

        The container is a separate widget that never becomes disabled itself,
        so its QSS ``:hover`` rule would otherwise still highlight the entry
        when the inner button is disabled. A ``nav_disabled`` property lets QSS
        hold the container background static while the entry cannot be used.
        """
        container.setProperty("nav_disabled", "false" if enabled else "true")
        style = container.style()
        if style is not None:
            style.unpolish(container)
            style.polish(container)

    def _set_button_hovered(self, button: QToolButton, hovered: bool) -> None:
        """Mirror the outer container's hover state onto its inner button.

        The container insets the button by 4px, so hovering that gap leaves the
        pointer outside the button and Qt's ``:hover`` never fires. Driving a
        ``hovered`` property from the container's enter/leave events lets QSS
        highlight the inner button across the whole container area. The state is
        never set on a disabled button, which shows a forbidden cursor instead.
        """
        button.setProperty("hovered", "true" if hovered else "false")
        style = button.style()
        if style is not None:
            style.unpolish(button)
            style.polish(button)

    def _enable_external_plot_trigger(self) -> None:
        plot_button = self._nav_buttons[CREATE_PLOT]
        tooltip = "Right click to open external window"
        plot_button.setToolTip(tooltip)
        # A QToolButton re-syncs its tooltip from its default action whenever the
        # action changes (e.g. when re-tinting the icon), so re-assert the custom
        # tooltip after Qt's internal sync runs.
        self._actions[CREATE_PLOT].changed.connect(
            lambda b=plot_button, t=tooltip: b.setToolTip(t)
        )
        self._plot_button = plot_button
        plot_button.installEventFilter(self)

    def _capture_status_button(self) -> None:
        self._status_button = self._nav_buttons[EXPERIMENT_STATUS]

    @override
    def eventFilter(self, obj: QObject | None, event: QEvent | None) -> bool:
        if (
            obj is self._plot_button
            and isinstance(event, QMouseEvent)
            and event.type() == QEvent.Type.MouseButtonPress
            and event.button() == Qt.MouseButton.RightButton
        ):
            self.external_plot_requested.emit()
        if event is not None and event.type() in {
            QEvent.Type.Enter,
            QEvent.Type.Leave,
        }:
            self._update_container_hover(obj, event.type() == QEvent.Type.Enter)
        return super().eventFilter(obj, event)

    def _update_container_hover(self, obj: QObject | None, entered: bool) -> None:
        button = next(
            (
                self._nav_buttons[name]
                for name, container in self._nav_containers.items()
                if container is obj
            ),
            None,
        )
        if button is None:
            return
        self._set_button_hovered(button, entered and button.isEnabled())

    def _create_nav_button(self, name: str, icon: QIcon) -> QToolButton:
        action = QAction(icon, name, self)
        action.setCheckable(True)
        action.setToolTip(name)
        action.triggered.connect(lambda _=False, n=name: self.page_requested.emit(n))
        self._action_group.addAction(action)
        self._actions[name] = action

        button = QToolButton(self)
        button.setDefaultAction(action)
        button.setObjectName(object_name_for_entry(name))
        button.setToolButtonStyle(self.toolButtonStyle())
        button.setIconSize(self.iconSize())
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        button.setFixedHeight(25)
        button.setProperty("hovered", "false")
        self._nav_buttons[name] = button
        action.changed.connect(lambda b=button: self._sync_button_cursor(b))
        action.toggled.connect(lambda _=False, n=name: self._tint_nav_icon(n))
        self._sync_button_cursor(button)
        return button

    @staticmethod
    def _sync_button_cursor(button: QToolButton) -> None:
        """Match ``button``'s cursor to its enabled state.

        An enabled button shows a pointing-hand cursor to signal it is
        clickable, while a disabled button shows a forbidden cursor. Disabled
        Qt widgets never receive hover events, so the cursor shape is the only
        affordance available to signal that a nav entry cannot be activated. Qt
        honours the cursor shape for a widget's screen region regardless of its
        enabled state, so the forbidden cursor displays even though the disabled
        button gets no mouse events.
        """
        if button.isEnabled():
            button.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            button.setCursor(Qt.CursorShape.ForbiddenCursor)

    def action_for(self, name: str) -> QAction:
        return self._actions[name]

    def _nav_icon_color(self, action: QAction) -> str:
        """Resolve the tint matching ``action``'s current QSS text colour."""
        if not action.isEnabled():
            token = NAV_COLOR_DISABLED_TOKEN
        elif action.isChecked():
            token = NAV_COLOR_CHECKED_TOKEN
        else:
            token = NAV_COLOR_DEFAULT_TOKEN
        return resolve_color(self._color_scheme, token)

    def _tint_nav_icon(self, name: str) -> None:
        """Re-tint a single nav icon to match its button's current text colour."""
        action = self._actions[name]
        icon_file = self._nav_icon_files[name]
        action.setIcon(load_icon(icon_file, color=self._nav_icon_color(action)))

    def _tint_all_nav_icons(self) -> None:
        for name in self._nav_icon_files:
            self._tint_nav_icon(name)

    def retint_all(self, color_scheme: ColorScheme) -> None:
        """Re-tint every icon for ``color_scheme`` after a runtime theme switch."""
        self._color_scheme = color_scheme
        self._tint_all_nav_icons()
        self._tint_collapse_icon()

    def _tint_collapse_icon(self) -> None:
        if self._collapse_action is None:
            return
        self._collapse_action.setIcon(
            load_icon(
                COLLAPSE_ICON,
                rotation=180 if self._collapsed else 0,
                color=resolve_color(self._color_scheme, COLLAPSE_COLOR_TOKEN),
            )
        )

    def set_current(self, name: str) -> None:
        self._actions[name].setChecked(True)

    def set_status_enabled(self, enabled: bool) -> None:
        self._actions[EXPERIMENT_STATUS].setEnabled(enabled)
        self._tint_nav_icon(EXPERIMENT_STATUS)

    def button_for(self, name: str) -> QToolButton | None:
        return self._nav_buttons.get(name)

    @property
    def collapsed(self) -> bool:
        return self._collapsed

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapsed = collapsed
        style = (
            Qt.ToolButtonStyle.ToolButtonIconOnly
            if collapsed
            else Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.setFixedWidth(COLLAPSED_WIDTH if collapsed else EXPANDED_WIDTH)
        self.setToolButtonStyle(style)
        for button in self._nav_buttons.values():
            button.setToolButtonStyle(style)
        if self._collapse_action is not None:
            label = EXPAND_SIDEBAR if collapsed else COLLAPSE_SIDEBAR
            self._collapse_action.setText(label)
            self._collapse_action.setToolTip(label)
            self._tint_collapse_icon()
        if self._title_label is not None:
            self._title_label.setVisible(not collapsed)
        self._keep_collapse_button_icon_only()
        self._apply_layout_for_collapse_state()

    def _apply_layout_for_collapse_state(self) -> None:
        """Tune header/nav spacing so expanded labels and collapsed icons align."""
        if self._header_layout is not None:
            self._header_layout.setContentsMargins(*_HEADER_MARGINS)
            self._header_layout.setSpacing(
                _HEADER_SPACING_COLLAPSED
                if self._collapsed
                else _HEADER_SPACING_EXPANDED
            )
        if self._nav_layout is not None:
            self._nav_layout.setContentsMargins(*_NAV_MARGINS)
            self._nav_layout.setSpacing(_NAV_SPACING)

    def _keep_collapse_button_icon_only(self) -> None:
        if self._collapse_button is not None:
            self._collapse_button.setToolButtonStyle(
                Qt.ToolButtonStyle.ToolButtonIconOnly
            )

    def _toggle_collapsed(self) -> None:
        self.set_collapsed(not self._collapsed)

    def add_status_entry(self, name: str) -> None:
        self._status_entries.append(name)
        entry_count = len(self._status_entries)
        if entry_count == 2:
            self._create_status_menu()
            for entry in self._status_entries:
                self._add_status_menu_item(entry)
        elif entry_count > 2:
            self._add_status_menu_item(name)

    def _create_status_menu(self) -> None:
        if self._status_button is None:
            return
        menu = QMenu(self._status_button)
        menu.setObjectName(STATUS_MENU_OBJECT_NAME)
        self._status_button.setMenu(menu)
        self._status_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu.aboutToHide.connect(self._on_status_menu_about_to_hide)

    def _add_status_menu_item(self, name: str) -> None:
        if self._status_button is None:
            return
        menu = self._status_button.menu()
        if menu is None:
            return
        action_list = menu.actions()
        act = QAction(text=name, parent=menu)
        act.triggered.connect(
            lambda _=False, n=name: self.status_entry_selected.emit(n)
        )
        act.triggered.connect(lambda _: self._mark_action_bold(menu, act))
        if action_list:
            menu.insertAction(action_list[0], act)
        else:
            menu.addAction(act)
        self._mark_action_bold(menu, menu.actions()[0])

    def _on_status_menu_about_to_hide(self) -> None:
        if self._status_button is not None:
            self._set_button_hovered(self._status_button, False)

    @staticmethod
    def _mark_action_bold(menu: QMenu, action_to_mark: QAction) -> None:
        for action in menu.actions():
            font = action.font()
            font.setBold(action is action_to_mark)
            action.setFont(font)
