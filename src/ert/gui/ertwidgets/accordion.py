from __future__ import annotations

from PyQt6.QtCore import QSize
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtWidgets import (
    QFrame,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ert.gui.detect_mode import is_dark_mode
from ert.gui.eds_colors import DARK_GREY, MEDIUM_GREEN
from ert.gui.icon_utils import load_stateful_icon

_ICON_SIZE = 24
# Qt draws 4px between icon and text; pad to the 16px EDS medium spacing.
_EXTRA_ICON_GAP = 12


class Accordion(QWidget):
    """A collapsible section styled after the EDS Accordion item.

    The header shows a chevron and the title; clicking it shows or hides
    the content.
    """

    toggled = Signal(bool)

    def __init__(
        self,
        title: str,
        content: QWidget,
        *,
        expanded: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        self.header = QPushButton(title)
        self.header.setObjectName("accordion_header")
        self.header.setCheckable(True)
        self.header.setFixedHeight(48)
        self.header.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.header.setIconSize(QSize(_ICON_SIZE + _EXTRA_ICON_GAP, _ICON_SIZE))
        icon_color = "white" if is_dark_mode() else MEDIUM_GREEN
        self._expand_icon = load_stateful_icon(
            "chevron_down.svg",
            resting=icon_color,
            active=icon_color,
            disabled=DARK_GREY,
            trailing_gap=_EXTRA_ICON_GAP / _ICON_SIZE,
        )
        self._collapse_icon = load_stateful_icon(
            "chevron_up.svg",
            resting=icon_color,
            active=icon_color,
            disabled=DARK_GREY,
            trailing_gap=_EXTRA_ICON_GAP / _ICON_SIZE,
        )

        self.content = QFrame()
        self.content.setObjectName("accordion_panel")
        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(16, 16, 16, 16)
        content_layout.addWidget(content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.header)
        layout.addWidget(self.content)

        self.header.setChecked(expanded)
        self.header.setIcon(self._collapse_icon if expanded else self._expand_icon)
        if not expanded:
            self.content.hide()
        self.header.toggled.connect(self._on_toggled)

    def _on_toggled(self, expanded: bool) -> None:
        self.header.setIcon(self._collapse_icon if expanded else self._expand_icon)
        self.content.setVisible(expanded)
        self.toggled.emit(expanded)
