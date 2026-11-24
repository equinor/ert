from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ert.gui.ertwidgets import ErtSummary, ParametersSummary
from ert.gui.ertwidgets.card import CARD_SPACING, create_card, create_card_title
from ert.gui.ertwidgets.chip import DataSource, SourceChip
from ert.gui.ertwidgets.models.ertsummary import ObservationCount

if TYPE_CHECKING:
    from ert.config import ErtConfig, ParameterConfig

logger = logging.getLogger(__name__)


class SummaryTemplate:
    def __init__(self, title: str, width: int = 150) -> None:
        super().__init__()

        self.text = ""
        self.__finished = False
        self.startGroup(title, width)

    def startGroup(self, title: str, width: int = 150) -> None:
        if not self.__finished:
            style = (
                f"display: inline-block; width: {width}px; "
                "vertical-align: top; float: left"
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

    def addColumn(self, header: str, items: list[Any]) -> None:
        if not self.__finished:
            header_style = "font-weight: bold; padding: 0px 10px 0px 5px;"
            row_style = "padding: 0px 10px 0px 5px;"
            cell = f'<div style="{header_style}">{header}</div>'
            for item in items:
                cell += f'<div style="{row_style}">{item}</div>'
            self.text += f'<td style="vertical-align: top;">{cell}</td>'

    def startColumns(self) -> None:
        if not self.__finished:
            self.text += "<table><tr>"

    def endColumns(self) -> None:
        if not self.__finished:
            self.text += "</tr></table>"

    def endGroup(self) -> None:
        if not self.__finished:
            self.text += "</div></br>\n"

    def getText(self) -> str:
        if not self.__finished:
            self.__finished = True
            self.endGroup()
        return f"<html>{self.text}</html>"


class SummaryPanel(QFrame):
    """Model summary: forward model, parameters and observations.

    Each card has a chip telling whether its content comes from the config
    file or from the ensemble selected in the experiment panel.
    """

    def __init__(self, config: ErtConfig) -> None:
        self.config = config
        QFrame.__init__(self)

        self.setMinimumWidth(250)
        self.setMinimumHeight(150)

        widget = QWidget(self)
        self._layout = QHBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(CARD_SPACING)
        widget.setLayout(self._layout)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        # A scroll bar would narrow the last card so that it no longer lines up
        # with the settings above; the cards' contents scroll on their own.
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(widget)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        title = create_card_title("Model summary")
        title.setObjectName("model_summary_title")
        layout.addWidget(title)
        layout.addWidget(scroll, 1)

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

        # Forward model steps always come from the config file, also when
        # continuing from a selected ensemble.
        self.forward_model_chip = self.addColumn(text.getText())

        self.parameters_summary = ParametersSummary()
        self.parameters_chip = self._add_card(self.parameters_summary)
        self.set_parameter_configuration(
            self.config.parameter_configurations_with_design_matrix
        )

        self._observations_label = self._create_text_label("")
        self.observations_chip = self._add_card(self._observations_label, stretch=False)
        self.set_observations(summary.getObservations())

    def set_parameter_configuration(
        self,
        parameter_configuration: list[ParameterConfig],
        source: DataSource = DataSource.CONFIG,
        ensemble_name: str | None = None,
    ) -> None:
        self.parameters_summary.set_parameters(parameter_configuration)
        self.parameters_chip.set_source(source, ensemble_name)

    def set_observations(
        self,
        observation_counts: list[ObservationCount],
        source: DataSource = DataSource.CONFIG,
        ensemble_name: str | None = None,
    ) -> None:
        text = SummaryTemplate(
            f"Observations ({sum(e['count'] for e in observation_counts)})"
        )
        for entry in observation_counts:
            text.addRow(f"{entry['observation_key']} ({entry['count']})")
        self._observations_label.setText(text.getText())
        self.observations_chip.set_source(source, ensemble_name)

    @staticmethod
    def _create_text_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.RichText)
        return label

    def addColumn(self, text: str) -> SourceChip:
        return self._add_card(self._create_text_label(text), stretch=False)

    def _add_card(self, widget: QWidget, *, stretch: bool = True) -> SourceChip:
        card = create_card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.addWidget(widget, 1 if stretch else 0)
        if not stretch:
            layout.addStretch(1)
        chip = SourceChip()
        layout.addWidget(chip, 0, Qt.AlignmentFlag.AlignLeft)

        self._layout.addWidget(card, 1)
        return chip

    def log_summary(self, run_model: str, num_realizations: int) -> None:
        summary = ErtSummary(self.config)

        observations = summary.getObservations()
        observations_count = sum(e["count"] for e in observations)

        _, _, parameter_count = summary.get_parameters()

        logger.info(
            f"Experiment summary:\n"
            f"Runmodel: {run_model}\n"
            f"Realizations: {num_realizations}\n"
            f"Parameters: {parameter_count}\n"
            f"Observations: {observations_count}"
        )

    @staticmethod
    def _runlength_encode_list(strings: list[str]) -> list[tuple[str, int]]:
        """Runlength encode a list of strings.

        Returns a list of tuples, first element is the string, and the second
        element is the count of consecutive occurrences of the string at the current
        position.
        """
        string_counts: list[tuple[str, int]] = []
        for string in strings:
            if not string_counts or string_counts[-1][0] != string:
                string_counts.append((string, 1))
            else:
                string_counts[-1] = (string, string_counts[-1][1] + 1)
        return string_counts
