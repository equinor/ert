import math
from collections.abc import Sequence
from datetime import UTC, datetime

from matplotlib import rcParams
from PyQt6.QtCore import QDateTime, QLocale, QTimeZone
from PyQt6.QtWidgets import (
    QAbstractButton,
    QComboBox,
    QDateTimeEdit,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

_MSECS_PER_DAY = 86_400_000


def _date_epoch_msecs() -> float:
    epoch = datetime.fromisoformat(str(rcParams["date.epoch"])).replace(tzinfo=UTC)
    return epoch.timestamp() * 1000


LimitsEdit = QLineEdit | QDateTimeEdit | QComboBox


class AxisLimitsDialog(QDialog):
    """Dialog for entering the minimum and maximum value of a plot axis.

    For date axes, the limits are given and returned as matplotlib date numbers.
    For categorical axes, given as (position, label) pairs sorted by position,
    the user picks the first and last category to show instead.
    """

    def __init__(
        self,
        axis_name: str,
        limits: tuple[float, float],
        *,
        is_date: bool = False,
        is_log: bool = False,
        categories: Sequence[tuple[float, str]] = (),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Set {axis_name}-axis limits")
        self.reset_requested = False
        self._is_date = is_date
        self._is_log = is_log

        minimum, maximum = sorted(limits)
        form = QFormLayout()
        self._minimum_edit: LimitsEdit
        self._maximum_edit: LimitsEdit
        if categories:
            visible = [
                index
                for index, (position, _) in enumerate(categories)
                if minimum <= position <= maximum
            ] or [0, len(categories) - 1]
            labels = [label for _, label in categories]
            self._minimum_edit = self._create_category_edit(labels, visible[0])
            self._maximum_edit = self._create_category_edit(labels, visible[-1])
            form.addRow("First:", self._minimum_edit)
            form.addRow("Last:", self._maximum_edit)
        else:
            self._minimum_edit = self._create_edit(minimum)
            self._maximum_edit = self._create_edit(maximum)
            form.addRow("Minimum:", self._minimum_edit)
            form.addRow("Maximum:", self._maximum_edit)

        self._error_label = QLabel()
        self._error_label.setStyleSheet("color: red;")

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Reset
        )
        reset_button = self._buttons.button(QDialogButtonBox.StandardButton.Reset)
        assert reset_button is not None
        reset_button.setToolTip("Restore the automatically computed limits")
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        self._buttons.clicked.connect(self._on_button_clicked)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._error_label)
        layout.addWidget(self._buttons)

        self._validate()

    def _create_edit(self, value: float) -> QLineEdit | QDateTimeEdit:
        if self._is_date:
            date_edit = QDateTimeEdit()
            date_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
            date_edit.setCalendarPopup(True)
            date_edit.setTimeZone(QTimeZone.utc())
            date_edit.setDateTime(
                QDateTime.fromMSecsSinceEpoch(
                    round(_date_epoch_msecs() + value * _MSECS_PER_DAY),
                    QTimeZone.utc(),
                )
            )
            date_edit.dateTimeChanged.connect(self._validate)
            return date_edit
        line_edit = QLineEdit(f"{value:.6g}")
        line_edit.textChanged.connect(self._validate)
        return line_edit

    def _create_category_edit(self, labels: list[str], index: int) -> QComboBox:
        combo_box = QComboBox()
        combo_box.addItems(labels)
        combo_box.setCurrentIndex(index)
        combo_box.currentIndexChanged.connect(self._validate)
        return combo_box

    def _value(self, edit: LimitsEdit) -> float | None:
        if isinstance(edit, QComboBox):
            return edit.currentIndex()
        if isinstance(edit, QDateTimeEdit):
            msecs = edit.dateTime().toMSecsSinceEpoch()
            return (msecs - _date_epoch_msecs()) / _MSECS_PER_DAY
        value, ok = QLocale.c().toDouble(edit.text().strip())
        return value if ok and math.isfinite(value) else None

    def limits(self) -> tuple[float, float]:
        """The entered limits, or the indices of the first and last category."""
        minimum = self._value(self._minimum_edit)
        maximum = self._value(self._maximum_edit)
        assert minimum is not None
        assert maximum is not None
        return minimum, maximum

    def _error_message(self) -> str:
        minimum = self._value(self._minimum_edit)
        maximum = self._value(self._maximum_edit)
        if minimum is None or maximum is None:
            return "Limits must be numbers"
        if isinstance(self._minimum_edit, QComboBox):
            return (
                "First category must not come after the last category"
                if minimum > maximum
                else ""
            )
        if minimum >= maximum:
            return "Minimum must be less than maximum"
        if self._is_log and minimum <= 0:
            return "Limits must be positive on a logarithmic axis"
        return ""

    def _validate(self) -> None:
        message = self._error_message()
        self._error_label.setText(message)
        self._error_label.setVisible(bool(message))
        ok_button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        assert ok_button is not None
        ok_button.setEnabled(not message)

    def _on_button_clicked(self, button: QAbstractButton) -> None:
        if self._buttons.buttonRole(button) == QDialogButtonBox.ButtonRole.ResetRole:
            self.reset_requested = True
            self.accept()

    @property
    def minimum_edit(self) -> LimitsEdit:
        return self._minimum_edit

    @property
    def maximum_edit(self) -> LimitsEdit:
        return self._maximum_edit
