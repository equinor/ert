from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QSignalBlocker, Qt
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ert.gui.ertwidgets import ErtSummary

if TYPE_CHECKING:
    from ert.config import ErtConfig

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
    selection_changed = Signal()

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

        forward_model_steps = self.config.forward_model_steps
        plural_s = ""
        if len(forward_model_steps) != 1:
            plural_s = "s"
        title = f"Forward model ({len(forward_model_steps):,} step{plural_s})"
        forward_model_list = summary.getForwardModels()
        text = SummaryTemplate(title)
        for fm_name, fm_count in self._runlength_encode_list(forward_model_list):
            text.addRow(f"{fm_name} x{fm_count}" if fm_count > 1 else fm_name)
        self.addForwardModelList(
            forward_model_steps,
            text.getText(),
        )

        parameter_list_updatable, parameter_list_not_updatable, parameter_count = (
            summary.get_parameters()
        )
        text = SummaryTemplate(f"Parameters ({parameter_count:,})", width=320)

        text.startColumns()
        text.addColumn("Updatable", parameter_list_updatable)
        text.addColumn("Non-updatable", parameter_list_not_updatable)
        text.endColumns()

        self.addColumn(text.getText())

        observation_counts = summary.getObservations()
        text = SummaryTemplate(
            f"Observations ({sum(e['count'] for e in observation_counts)})"
        )
        for entry in observation_counts:
            text.addRow(f"{entry['observation_key']} ({entry['count']})")

        self.addColumn(text.getText())

    def addForwardModelList(self, steps: list[Any], summary_text: str) -> None:
        self.forward_model_summary = QLabel(summary_text)
        self.forward_model_summary.setWordWrap(True)
        self.forward_model_summary.setTextFormat(Qt.TextFormat.RichText)
        summary_layout = QVBoxLayout()
        summary_layout.addWidget(self.forward_model_summary)

        self.forward_model_selection = QWidget()
        selection_layout = QVBoxLayout(self.forward_model_selection)
        selection_layout.setContentsMargins(0, 0, 0, 0)
        buttons = QHBoxLayout()
        check_all = QPushButton("Check all")
        check_all.setObjectName("check_all_forward_model_steps")
        check_all.clicked.connect(lambda: self._set_all_steps_checked(True))
        buttons.addWidget(check_all)

        uncheck_all = QPushButton("Uncheck all")
        uncheck_all.setObjectName("uncheck_all_forward_model_steps")
        uncheck_all.clicked.connect(lambda: self._set_all_steps_checked(False))
        buttons.addWidget(uncheck_all)

        invert = QPushButton("Invert")
        invert.setObjectName("invert_forward_model_steps")
        invert.clicked.connect(self._invert_step_selection)
        buttons.addWidget(invert)

        selection_layout.addLayout(buttons)
        self.forward_model_step_list = QListWidget()
        self.forward_model_step_list.setObjectName("forward_model_step_list")
        for index, step in enumerate(steps, start=1):
            arguments = ", ".join(
                f"{key}={value}" for key, value in step.private_args.items()
            )
            text = f"{index}: {step.name}"
            if arguments:
                text += f"({arguments})"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setToolTip(text)
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
            )
            item.setCheckState(Qt.CheckState.Checked)
            self.forward_model_step_list.addItem(item)
        self.forward_model_step_list.itemChanged.connect(
            lambda _: self.selection_changed.emit()
        )
        self.forward_model_step_list.setMaximumHeight(100)
        selection_layout.addWidget(self.forward_model_step_list)
        self.forward_model_selection.hide()
        summary_layout.addWidget(self.forward_model_selection)
        summary_layout.addStretch(1)
        self._layout.addLayout(summary_layout)

    def set_step_selection_visible(self, visible: bool) -> None:
        self.forward_model_summary.setVisible(not visible)
        self.forward_model_selection.setVisible(visible)

    def selected_step_indices(self) -> list[int]:
        selected: list[int] = []
        for row in range(self.forward_model_step_list.count()):
            item = self.forward_model_step_list.item(row)
            assert item is not None
            if item.checkState() == Qt.CheckState.Checked:
                selected.append(item.data(Qt.ItemDataRole.UserRole))
        return selected

    def _set_all_steps_checked(self, checked: bool) -> None:
        blocker = QSignalBlocker(self.forward_model_step_list)
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for row in range(self.forward_model_step_list.count()):
            item = self.forward_model_step_list.item(row)
            assert item is not None
            item.setCheckState(state)
        blocker.unblock()
        self.selection_changed.emit()

    def _invert_step_selection(self) -> None:
        blocker = QSignalBlocker(self.forward_model_step_list)
        for row in range(self.forward_model_step_list.count()):
            item = self.forward_model_step_list.item(row)
            assert item is not None
            item.setCheckState(
                Qt.CheckState.Unchecked
                if item.checkState() == Qt.CheckState.Checked
                else Qt.CheckState.Checked
            )
        blocker.unblock()
        self.selection_changed.emit()

    def addColumn(self, text: str) -> None:
        layout = QVBoxLayout()
        text_widget = QLabel(text)
        text_widget.setWordWrap(True)
        text_widget.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(text_widget)
        layout.addStretch(1)

        self._layout.addLayout(layout)

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
        string_counts: list[tuple[str, int]] = []
        for string in strings:
            if not string_counts or string_counts[-1][0] != string:
                string_counts.append((string, 1))
            else:
                string_counts[-1] = (string, string_counts[-1][1] + 1)
        return string_counts
