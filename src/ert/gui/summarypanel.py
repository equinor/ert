from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ert.gui.ertwidgets.models.ertsummary import ErtSummary

if TYPE_CHECKING:
    from ert.config import ErtConfig


class SummaryTemplate:
    def __init__(self, title: str) -> None:
        super().__init__()

        self.text = ""
        self.__finished = False
        self.startGroup(title)

    def startGroup(self, title: str) -> None:
        if not self.__finished:
            style = (
                "display: inline-block; width: 150px; vertical-align: top; float: left"
            )
            self.text += f'<div style="{style}">\n'
            self.addTitle(title)

    def addTitle(self, title: str) -> None:
        if not self.__finished:
            style = "font-size: 16px; font-weight: bold;"
            self.text += f'<div style="{style}">{title}</div>'

    def addRow(self, value: Any) -> None:
        if not self.__finished:
            style = "text-indent: 5px;"
            self.text += f'<div style="{style}">{value}</div>'

    def endGroup(self) -> None:
        if not self.__finished:
            self.text += "</div></br>\n"

    def getText(self) -> str:
        if not self.__finished:
            self.__finished = True
            self.endGroup()
        return f"<html>{self.text}</html>"


class SummaryPanel(QFrame):
    def __init__(self, config: ErtConfig) -> None:
        self.config = config
        QFrame.__init__(self)

        self.setMinimumWidth(250)
        self.setMinimumHeight(150)

        widget = QWidget(self)
        self._layout = QHBoxLayout()
        widget.setLayout(self._layout)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

        self.setLayout(layout)
        self.updateSummary()

    def updateSummary(self) -> None:
        summary = ErtSummary(self.config)

        forward_model_list = summary.getForwardModels()
        plural_s = ""
        if not len(forward_model_list) or len(forward_model_list) > 1:
            plural_s = "s"
        text = SummaryTemplate(
            f"Forward model ({len(forward_model_list):,} step{plural_s})"
        )
        for fm_name, fm_count in self._runlength_encode_list(forward_model_list):
            if fm_count == 1:
                text.addRow(fm_name)
            else:
                text.addRow(f"{fm_name} x{fm_count}")

        self.addColumn(text.getText())

        parameter_list, parameter_count = summary.getParameters()
        text = SummaryTemplate(f"Parameters ({parameter_count:,})")
        for parameters in parameter_list:
            text.addRow(parameters)

        self.addColumn(text.getText())

        observation_counts = summary.getObservations()
        text = SummaryTemplate(
            f"Observations ({sum(e['count'] for e in observation_counts)})"
        )
        for entry in observation_counts:
            text.addRow(f"{entry['observation_key']} ({entry['count']})")

        self.addColumn(text.getText())

    def addColumn(self, text: str) -> None:
        layout = QVBoxLayout()
        text_widget = QLabel(text)
        text_widget.setWordWrap(True)
        text_widget.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(text_widget)
        layout.addStretch(1)

        self._layout.addLayout(layout)

    @staticmethod
    def _runlength_encode_list(strings: list[str]) -> list[tuple[str, int]]:
        """Runlength encode a list of strings.

        Returns a list of tuples, first element is the string, and the second
        element is the count of consecutive occurences of the string at the current
        position."""
        string_counts: list[tuple[str, int]] = []
        for string in strings:
            if not string_counts or string_counts[-1][0] != string:
                string_counts.append((string, 1))
            else:
                string_counts[-1] = (string, string_counts[-1][1] + 1)
        return string_counts
