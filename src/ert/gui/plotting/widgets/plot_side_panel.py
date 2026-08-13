from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtWidgets import (
    QWIDGETSIZE_MAX,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ert.gui.detect_mode import is_dark_mode
from ert.gui.icon_utils import load_icon

SIDE_PANEL_BACKGROUND_DARK = "rgb(64, 64, 64)"
SIDE_PANEL_BACKGROUND_LIGHT = "lightgray"
SIDE_PANEL_BORDER_DARK = "#2d2d2d"
SIDE_PANEL_BORDER_LIGHT = "#b0b0b0"


class PlotSidePanel(QDockWidget):
    def __init__(
        self,
        title: str,
        content: QWidget,
        main_window: QMainWindow,
        min_expanded_width: int = 250,
        *,
        on_right: bool = False,
    ) -> None:
        super().__init__(main_window)
        self._main_window = main_window
        self._content = content
        self._expanded_width = min_expanded_width
        self._min_expanded_width = min_expanded_width
        self._title = title

        side = "right" if on_right else "left"
        self.setObjectName(f"{side}_plot_side_panel")
        self.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)

        self._body = QWidget()
        self._body.setObjectName("plot_side_panel_body")
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.addWidget(content)
        self.setWidget(self._body)

        self._toggle_button = QToolButton()
        self._toggle_button.setCheckable(True)
        self._toggle_button.setChecked(True)
        self._toggle_button.setAutoRaise(True)
        self._toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_button.setToolTip(f"Collapse {self._title} panel")
        self._collapse_icon = load_icon(f"{side}_panel_close.svg")
        self._expand_icon = load_icon(f"{side}_panel_open.svg")

        self._toggle_button.setIconSize(QSize(24, 24))
        self._toggle_button.setIcon(self._collapse_icon)

        self._title_label = QLabel(title)

        title_bar = QWidget()
        title_bar.setObjectName("plot_side_panel_titlebar")
        background = (
            SIDE_PANEL_BACKGROUND_DARK
            if is_dark_mode()
            else SIDE_PANEL_BACKGROUND_LIGHT
        )
        border_color = (
            SIDE_PANEL_BORDER_DARK if is_dark_mode() else SIDE_PANEL_BORDER_LIGHT
        )
        divider_edge = "left" if on_right else "right"
        self.setStyleSheet(
            "QWidget#plot_side_panel_titlebar, QWidget#plot_side_panel_body {"
            f"  background-color: {background};"
            "}"
            "QWidget#plot_side_panel_titlebar {"
            f"  border-bottom: 1px solid {border_color};"
            "}"
            "QWidget#plot_side_panel_body {"
            f"  border-{divider_edge}: 1px solid {border_color};"
            "}"
        )
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(2, 2, 2, 2)
        title_layout.setSpacing(2)
        # Layout order is important here
        # Places button either on left or right side
        # based on the order
        if on_right:
            title_layout.addWidget(self._toggle_button)
        title_layout.addStretch(1)
        title_layout.addWidget(self._title_label)
        title_layout.addStretch(1)
        if not on_right:
            title_layout.addWidget(self._toggle_button)
        self.setTitleBarWidget(title_bar)
        self._toggle_button.toggled.connect(self._set_expanded)

    def _set_expanded(self, expanded: bool) -> None:
        if not expanded:
            self._expanded_width = self.width()

        self._content.setVisible(expanded)
        self._title_label.setVisible(expanded)
        self._toggle_button.setIcon(
            self._collapse_icon if expanded else self._expand_icon
        )
        self._toggle_button.setToolTip(
            f"{'Collapse' if expanded else 'Open'} {self._title} panel"
        )

        if expanded:

            def _expand() -> None:
                self.setMinimumWidth(self._min_expanded_width)
                self.setMaximumWidth(QWIDGETSIZE_MAX)
                self._main_window.resizeDocks(
                    [self], [self._expanded_width], Qt.Orientation.Horizontal
                )

            QTimer.singleShot(0, _expand)
        else:
            QTimer.singleShot(
                0,
                lambda: self.setFixedWidth(self._toggle_button.sizeHint().width() + 8),
            )
