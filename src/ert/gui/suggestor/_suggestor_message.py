from __future__ import annotations

from typing import Any, Self

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtWidgets import (
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ert.gui import is_dark_mode

from ..ertwidgets.copyablelabel import _CopyButton
from ._colors import (
    BLUE_BACKGROUND,
    BLUE_TEXT,
    RED_BACKGROUND,
    RED_TEXT,
    YELLOW_BACKGROUND,
    YELLOW_TEXT,
)


def _svg_icon(image_name: str) -> QSvgWidget:
    widget = QSvgWidget(f"img:{image_name}.svg")
    widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return widget


class SuggestorMessage(QWidget):
    def __init__(
        self,
        header: str,
        header_text_color: str,
        bg_color: str,
        icon: QWidget,
        message: str,
        locations: list[str],
    ) -> None:
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setStyleSheet(
            f"""
            background-color: {bg_color};
            border-radius: 4px;
        """
        )
        shadowEffect = QGraphicsDropShadowEffect()
        shadowEffect.setColor(QColor.fromRgb(0, 0, 0, int(0.12 * 255)))
        shadowEffect.setBlurRadius(5)
        shadowEffect.setXOffset(2)
        shadowEffect.setYOffset(4)
        self.setGraphicsEffect(shadowEffect)
        self.setContentsMargins(0, 0, 0, 0)

        self._icon = icon
        self._message = message.replace("<", "&lt;").replace(">", "&gt;")
        self._locations = locations

        if self._locations and not self._locations[0]:
            self._locations.pop(0)

        self._header = header
        self._header_text_color = header_text_color

        self._text_color = self.palette().text().color().name()
        if is_dark_mode():
            self._text_color = "#303030"

        self._hbox = QHBoxLayout()
        self._hbox.setContentsMargins(16, 16, 16, 16)
        self._hbox.addWidget(self._icon, alignment=Qt.AlignmentFlag.AlignTop)
        self.setLayout(self._hbox)

        self.lbl = QLabel(self._collapsed_text())
        self.lbl.setOpenExternalLinks(False)
        self.lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl.setWordWrap(True)
        self._expanded = False
        if len(self._locations) > 1:
            self._expand_collapse_label = QLabel(self._expand_link())
            self._expand_collapse_label.setOpenExternalLinks(False)
            self._expand_collapse_label.linkActivated.connect(self._toggle_expand)

            self._vbox = QWidget()
            layout = QVBoxLayout()
            self._vbox.setLayout(layout)
            layout.addWidget(self.lbl)
            layout.addWidget(self._expand_collapse_label)
            self._hbox.addWidget(self._vbox, alignment=Qt.AlignmentFlag.AlignTop)
        else:
            self._expand_collapse_label = QLabel()
            self._hbox.addWidget(self.lbl, alignment=Qt.AlignmentFlag.AlignTop)
        self._hbox.addWidget(
            _CopyButton(QLabel(message)), alignment=Qt.AlignmentFlag.AlignTop
        )

    def _toggle_expand(self, _link: Any) -> None:
        if self._expanded:
            self.lbl.setText(self._collapsed_text())
            self._expand_collapse_label.setText(self._expand_link())
        else:
            self.lbl.setText(self._expanded_text())
            self._expand_collapse_label.setText(self._hide_link())
        self._expanded = not self._expanded

    def _hide_link(self) -> str:
        return " <a href=#morelocations>show less</a>"

    def _expand_link(self) -> str:
        return f" <a href=#morelocations>and {len(self._locations) - 1} more</a>"

    def _collapsed_text(self) -> str:
        location_paragraph = ""
        if self._locations:
            location_paragraph = (
                "<p>"
                + self._color_bold("location: ")
                + f"<div style='color: {self._text_color};'>{self._locations[0]}"
                + "</div></p>"
            )

        return self._text(location_paragraph)

    def _expanded_text(self) -> str:
        location_paragraphs = ""
        first = True
        for loc in self._locations:
            if first:
                location_paragraphs += (
                    f"<p>{self._color_bold('location:')}"
                    f"<div style='color: {self._text_color};'>"
                    f"{loc}</div></p>"
                )
                first = False
            else:
                location_paragraphs += (
                    f"<p style='color: {self._text_color};'>{loc}</p>"
                )

        return self._text(location_paragraphs)

    def _text(self, location: str) -> str:
        return (
            '<div style="font-size: 16px; line-height: 24px;">'
            + self._color_bold(self._header)
            + f"<p style='white-space: pre-wrap; color: {self._text_color};'>"
            + f"{self._message}</p>"
            + location
            + "</div>"
        )

    def _color_bold(self, text: str) -> str:
        return f'<b style="color: {self._header_text_color}">{text}</b>'

    @classmethod
    def error_msg(cls, message: str, locations: list[str]) -> Self:
        return cls(
            "Error: ", RED_TEXT, RED_BACKGROUND, _svg_icon("error"), message, locations
        )

    @classmethod
    def warning_msg(cls, message: str, locations: list[str]) -> Self:
        return cls(
            "Warning: ",
            YELLOW_TEXT,
            YELLOW_BACKGROUND,
            _svg_icon("warning"),
            message,
            locations,
        )

    @classmethod
    def deprecation_msg(cls, message: str, locations: list[str]) -> Self:
        return cls(
            "Deprecation: ",
            BLUE_TEXT,
            BLUE_BACKGROUND,
            _svg_icon("bell"),
            message,
            locations,
        )
