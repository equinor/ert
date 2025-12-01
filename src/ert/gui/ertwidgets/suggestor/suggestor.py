from __future__ import annotations

import functools
import logging
import webbrowser
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ert.gui import is_dark_mode

from ._colors import BLUE_TEXT
from ._suggestor_message import SuggestorMessage

if TYPE_CHECKING:
    from ert.config import ErrorInfo, WarningInfo

logger = logging.getLogger(__name__)


def _clicked_help_button(menu_label: str, link: str) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"Gui utility: {menu_label} help link was used from Suggestor")
    webbrowser.open(link)


def _combine_messages(infos: Sequence[ErrorInfo]) -> list[tuple[str, list[str]]]:
    combined = defaultdict(list)
    for info in infos:
        combined[info.message].append(info.location())
    return list(combined.items())


LIGHT_GREY = "#f7f7f7"
MEDIUM_GREY = "#eaeaea"
HEAVY_GREY = "#dcdcdc"
DARK_GREY = "#bebebe"
LIGHT_GREEN = "#deedee"
MEDIUM_GREEN = "#007079"
DARK_GREEN = "#004f55"

BUTTON_STYLE = f"""
QPushButton {{
    background-color: {MEDIUM_GREEN};
    color: white;
    border-radius: 4px;
    border: 2px solid {MEDIUM_GREEN};
    height: 36px;
    padding: 0px 16px 0px 16px;
}}
QPushButton:hover {{
    background: {DARK_GREEN};
    border: 2px solid {DARK_GREEN};
}};
"""

LINK_STYLE = f"""
QPushButton {{
    color: {BLUE_TEXT};
    border: 0px solid white;
    margin-left: 34px;
    height: 36px;
    padding: 0px 12px 0px 12px;
    text-decoration: underline;
    text-align: left;
    font-size: 16px;
    padding: 0px;
}}
"""

DISABLED_BUTTON_STYLE = f"""
    background-color: {MEDIUM_GREY};
    color: {DARK_GREY};
    border-radius: 4px;
    border: 2px solid {MEDIUM_GREY};
    height: 36px;
    padding: 0px 16px 0px 16px;
"""

SECONDARY_BUTTON_STYLE = f"""
QPushButton {{
    background-color: {LIGHT_GREY};
    color: {MEDIUM_GREEN};
    border-radius: 4px;
    border: 2px solid {MEDIUM_GREEN};
    height: 36px;
    padding: 0px 16px 0px 16px;
}}
QPushButton:hover {{
    background-color: {LIGHT_GREEN};
}};
"""


class Suggestor(QWidget):
    def __init__(
        self,
        errors: list[ErrorInfo] | None = None,
        warnings: list[WarningInfo] | None = None,
        deprecations: list[WarningInfo] | None = None,
        continue_action: Callable[[], None] | None = None,
        help_links: dict[str, str] | None = None,
        widget_info: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent, flags=Qt.WindowType.Window)
        if errors is None:
            errors = []
        if warnings is None:
            warnings = []
        if deprecations is None:
            deprecations = []
        self._continue_action = continue_action
        self.setWindowTitle("ERT")

        self.__layout = QVBoxLayout()
        self.setLayout(self.__layout)
        self.__layout.setContentsMargins(32, 16, 32, 16)
        self.__layout.addWidget(QLabel(widget_info))
        self.__layout.addSpacing(20)

        if not is_dark_mode():
            self.setStyleSheet(f"background-color: {LIGHT_GREY};")

        data_widget = QWidget(parent=self)
        data_layout = QHBoxLayout()
        data_widget.setLayout(data_layout)
        data_layout.setSpacing(16)
        data_layout.setContentsMargins(0, 0, 0, 0)
        data_layout.addWidget(self._problem_area(errors, warnings, deprecations))
        if help_links:
            data_layout.addWidget(self._help_panel(help_links))
        self.__layout.addWidget(data_widget)

        self.__layout.addWidget(self._action_buttons())

    def _help_panel(self, help_links: dict[str, str]) -> QFrame:
        help_button_frame = QFrame(parent=self)
        help_button_frame.setContentsMargins(0, 0, 0, 0)
        help_button_frame.setStyleSheet(
            f"""
            background-color: {MEDIUM_GREY};
            border-radius: 4px;
            border: 2px solid {HEAVY_GREY};
            """
        )
        help_button_frame.setMinimumWidth(388)
        help_button_frame.setMaximumWidth(388)
        help_buttons_layout = QVBoxLayout()
        help_buttons_layout.setContentsMargins(0, 30, 20, 20)
        help_button_frame.setLayout(help_buttons_layout)

        help_header = QLabel("Helpful links", parent=self)
        help_header.setContentsMargins(0, 0, 0, 0)
        help_header.setStyleSheet(
            f"font-size: 24px; color: {BLUE_TEXT}; border: none; margin-left: 30px;"
        )
        help_buttons_layout.addWidget(help_header, alignment=Qt.AlignmentFlag.AlignTop)

        separator = QFrame(parent=self)
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet(f"color: {HEAVY_GREY};")
        separator.setFixedWidth(388)
        help_buttons_layout.addWidget(separator)

        for menu_label, link in help_links.items():
            button = QPushButton(menu_label, parent=self)
            button.setStyleSheet(LINK_STYLE)
            button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            button.setObjectName(menu_label)
            button.clicked.connect(
                functools.partial(_clicked_help_button, menu_label, link)
            )
            help_buttons_layout.addWidget(button)

        help_buttons_layout.addStretch(-1)

        return help_button_frame

    def _problem_area(
        self,
        errors: list[ErrorInfo],
        warnings: list[WarningInfo],
        deprecations: list[WarningInfo],
    ) -> QWidget:
        problem_area = QWidget(parent=self)
        problem_area.setContentsMargins(0, 0, 0, 0)
        area_layout = QVBoxLayout()
        problem_area.setLayout(area_layout)
        area_layout.setContentsMargins(0, 0, 0, 0)
        area_layout.addWidget(self._messages(errors, warnings, deprecations))
        return problem_area

    def _action_buttons(self) -> QWidget:
        def run_pressed() -> None:
            assert self._continue_action
            self._continue_action()
            self.close()

        buttons = QWidget(parent=self)
        buttons_layout = QHBoxLayout()
        buttons_layout.insertStretch(-1, -1)
        buttons_layout.setContentsMargins(0, 24, 0, 0)

        give_up = QPushButton("Close")
        give_up.setObjectName("close_button")
        give_up.setStyleSheet(BUTTON_STYLE)
        give_up.pressed.connect(self.close)

        if self._continue_action:
            run = QPushButton("Open ERT")
            run.setStyleSheet(BUTTON_STYLE)
            run.setObjectName("run_ert_button")
            run.pressed.connect(run_pressed)
            buttons_layout.addWidget(run)

            give_up.setStyleSheet(SECONDARY_BUTTON_STYLE)

        buttons_layout.addWidget(give_up)
        buttons.setLayout(buttons_layout)

        return buttons

    def _messages(
        self,
        errors: Sequence[ErrorInfo],
        warnings: Sequence[WarningInfo],
        deprecations: Sequence[WarningInfo],
    ) -> QScrollArea:
        CARD_WIDTH = 450
        CARD_HEIGHT = 220
        PADDING = 24
        NUM_COLUMNS = 2

        suggest_msgs = QWidget(parent=self)
        suggest_msgs.setObjectName("suggestor_messages")
        suggest_msgs.setContentsMargins(0, 0, 16, 0)
        suggest_layout = QGridLayout()
        suggest_layout.setContentsMargins(0, 0, 0, 0)
        suggest_layout.setColumnMinimumWidth(0, CARD_WIDTH)
        suggest_layout.setSpacing(PADDING)

        column = 0
        row = 0
        num = 0
        for messages, message_type in [
            (errors, SuggestorMessage.error_msg),
            (warnings, SuggestorMessage.warning_msg),
            (deprecations, SuggestorMessage.deprecation_msg),
        ]:
            for combined in _combine_messages(messages):
                suggest_layout.addWidget(message_type(*combined), row, column)
                if column:
                    row += 1
                column = (column + 1) % NUM_COLUMNS
                num += 1
        suggest_layout.setRowStretch(row + 1, 1)

        width = 1440
        if num <= 1:
            width -= CARD_WIDTH

        height = 1024
        if row < 4:
            height -= (4 - (row + column)) * CARD_HEIGHT
        self.resize(width, height)

        suggest_msgs.setLayout(suggest_layout)
        scroll = QScrollArea()
        scroll.setStyleSheet(
            f"""
            QScrollArea {{
                border: none;
                width: 128px;
            }}
            QScrollBar {{
                border: none;
                background-color: {LIGHT_GREY};
                width: 10px;
            }}
            QScrollBar::handle {{
                border: none;
                background-color: {HEAVY_GREY};
                border-radius: 4px;
            }}
            QScrollBar::sub-line:vertical, QScrollBar::add-line:vertical {{
                background: none;
            }}
        """
        )
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setWidget(suggest_msgs)
        scroll.setContentsMargins(0, 0, 0, 0)

        scroll.setMinimumWidth(min(2, num) * (CARD_WIDTH + PADDING))
        scroll.setMinimumHeight(CARD_HEIGHT + PADDING)
        return scroll
