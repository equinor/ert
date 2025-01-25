from __future__ import annotations

from copy import copy
from datetime import date
from typing import TYPE_CHECKING, Any, ClassVar, cast

from PyQt6.QtGui import QDoubleValidator, QIntValidator
from PyQt6.QtWidgets import QLabel, QLineEdit, QStackedWidget

from ert.gui.tools.plot.plottery import PlotContext, PlotLimits
from ert.gui.tools.plot.widgets import ClearableLineEdit, CustomDateEdit

from .customization_view import CustomizationView

if TYPE_CHECKING:
    from ert.gui.tools.plot.plottery import PlotConfig


class StackedInput(QStackedWidget):
    def __init__(self) -> None:
        QStackedWidget.__init__(self)
        self._inputs: dict[str | None, QLineEdit | QLabel | CustomDateEdit] = {}
        self._index_map: dict[str | None, int] = {}
        self.addInput(PlotContext.UNKNOWN_AXIS, QLabel("Fixed"))
        self._current_name: str | None = PlotContext.UNKNOWN_AXIS

    def addInput(
        self, name: str | None, widget: QLineEdit | QLabel | CustomDateEdit
    ) -> None:
        index = self.addWidget(widget)
        self._inputs[name] = widget
        self._index_map[name] = index

    def switchToInput(self, name: str | None) -> None:
        index_for_name = self._index_map[name]
        self.setCurrentIndex(index_for_name)
        self._current_name = name


class LimitsStack(StackedInput):
    FLOAT_AXIS: ClassVar[list[str]] = [
        PlotContext.VALUE_AXIS,
        PlotContext.DENSITY_AXIS,
    ]
    INT_AXIS: ClassVar[list[str]] = [PlotContext.INDEX_AXIS, PlotContext.COUNT_AXIS]
    NUMBER_AXIS = FLOAT_AXIS + INT_AXIS

    def __init__(self) -> None:
        StackedInput.__init__(self)
        self.addInput(
            PlotContext.COUNT_AXIS,
            self.createIntegerLineEdit(minimum=0, placeholder="Default value"),
        )
        self.addInput(PlotContext.DATE_AXIS, CustomDateEdit())
        self.addInput(
            PlotContext.DENSITY_AXIS,
            self.createDoubleLineEdit(minimum=0, placeholder="Default value"),
        )
        self.addInput(
            PlotContext.INDEX_AXIS,
            self.createIntegerLineEdit(minimum=0, placeholder="Default value"),
        )
        self.addInput(
            PlotContext.VALUE_AXIS,
            self.createDoubleLineEdit(placeholder="Default value"),
        )

    @staticmethod
    def createDoubleLineEdit(
        minimum: float | None = None,
        maximum: float | None = None,
        placeholder: str = "",
    ) -> ClearableLineEdit:
        line_edit = ClearableLineEdit(placeholder=placeholder)
        validator = QDoubleValidator()

        if minimum is not None:
            validator.setBottom(minimum)

        if maximum is not None:
            validator.setTop(maximum)

        line_edit.setValidator(validator)
        return line_edit

    @staticmethod
    def createIntegerLineEdit(
        minimum: int | None = None,
        maximum: int | None = None,
        placeholder: str = "",
    ) -> ClearableLineEdit:
        line_edit = ClearableLineEdit(placeholder=placeholder)
        validator = QIntValidator()

        if minimum is not None:
            validator.setBottom(minimum)

        if maximum is not None:
            validator.setTop(maximum)

        line_edit.setValidator(validator)
        return line_edit

    def setValue(self, axis_name: str | None, value: Any) -> None:
        input_ = self._inputs[axis_name]

        if axis_name in LimitsStack.NUMBER_AXIS and (
            issubclass(type(input_), QLineEdit) or issubclass(type(input_), QLabel)
        ):
            input_ = cast(QLineEdit | QLabel, input_)
            input_.setText(str(value) if value is not None else "")
        elif axis_name == PlotContext.DATE_AXIS and type(input_) is CustomDateEdit:
            input_.setDate(value)

    def getValue(self, axis_name: str | None) -> float | int | date | None:
        input_ = self._inputs[axis_name]
        result: float | int | date | None = None
        if issubclass(type(input_), QLineEdit) or issubclass(type(input_), QLabel):
            try:
                input_ = cast(QLineEdit | QLabel, input_)
                if axis_name in LimitsStack.FLOAT_AXIS:
                    result = float(input_.text())
                elif axis_name in LimitsStack.INT_AXIS:
                    result = int(input_.text())
            except ValueError:
                result = None
        elif axis_name == PlotContext.DATE_AXIS and type(input_) is CustomDateEdit:
            result = input_.date()

        return result


class LimitsWidget:
    def __init__(self) -> None:
        self._limits = PlotLimits()
        self._x_minimum_stack = LimitsStack()
        self._x_maximum_stack = LimitsStack()
        self._x_current_input_name: str | None = PlotContext.UNKNOWN_AXIS

        self._y_minimum_stack = LimitsStack()
        self._y_maximum_stack = LimitsStack()
        self._y_current_input_name: str | None = PlotContext.UNKNOWN_AXIS

    @property
    def x_minimum_stack(self) -> LimitsStack:
        return self._x_minimum_stack

    @property
    def x_maximum_stack(self) -> LimitsStack:
        return self._x_maximum_stack

    @property
    def y_minimum_stack(self) -> LimitsStack:
        return self._y_minimum_stack

    @property
    def y_maximum_stack(self) -> LimitsStack:
        return self._y_maximum_stack

    @property
    def limits(self) -> PlotLimits:
        self._updateLimits()
        return copy(self._limits)

    @limits.setter
    def limits(self, value: PlotLimits) -> None:
        self._limits = copy(value)
        self._updateWidgets()

    def _updateWidgets(self) -> None:
        limits = self._limits
        self._x_minimum_stack.setValue(PlotContext.DATE_AXIS, limits.date_minimum)
        self._x_maximum_stack.setValue(PlotContext.DATE_AXIS, limits.date_maximum)
        self._y_minimum_stack.setValue(PlotContext.DATE_AXIS, limits.date_minimum)
        self._y_maximum_stack.setValue(PlotContext.DATE_AXIS, limits.date_maximum)

        self._x_minimum_stack.setValue(PlotContext.DENSITY_AXIS, limits.density_minimum)
        self._x_maximum_stack.setValue(PlotContext.DENSITY_AXIS, limits.density_maximum)
        self._y_minimum_stack.setValue(PlotContext.DENSITY_AXIS, limits.density_minimum)
        self._y_maximum_stack.setValue(PlotContext.DENSITY_AXIS, limits.density_maximum)

        self._x_minimum_stack.setValue(PlotContext.COUNT_AXIS, limits.count_minimum)
        self._x_maximum_stack.setValue(PlotContext.COUNT_AXIS, limits.count_maximum)
        self._y_minimum_stack.setValue(PlotContext.COUNT_AXIS, limits.count_minimum)
        self._y_maximum_stack.setValue(PlotContext.COUNT_AXIS, limits.count_maximum)

        self._x_minimum_stack.setValue(PlotContext.INDEX_AXIS, limits.index_minimum)
        self._x_maximum_stack.setValue(PlotContext.INDEX_AXIS, limits.index_maximum)
        self._y_minimum_stack.setValue(PlotContext.INDEX_AXIS, limits.index_minimum)
        self._y_maximum_stack.setValue(PlotContext.INDEX_AXIS, limits.index_maximum)

        self._x_minimum_stack.setValue(PlotContext.VALUE_AXIS, limits.value_minimum)
        self._x_maximum_stack.setValue(PlotContext.VALUE_AXIS, limits.value_maximum)
        self._y_minimum_stack.setValue(PlotContext.VALUE_AXIS, limits.value_minimum)
        self._y_maximum_stack.setValue(PlotContext.VALUE_AXIS, limits.value_maximum)

    def _updateLimits(self) -> None:
        if self._x_current_input_name is not PlotContext.UNKNOWN_AXIS:
            minimum = self._x_minimum_stack.getValue(self._x_current_input_name)
            maximum = self._x_maximum_stack.getValue(self._x_current_input_name)
            self._updateLimit(self._x_current_input_name, minimum, maximum)

        if self._y_current_input_name is not PlotContext.UNKNOWN_AXIS:
            minimum = self._y_minimum_stack.getValue(self._y_current_input_name)
            maximum = self._y_maximum_stack.getValue(self._y_current_input_name)
            self._updateLimit(self._y_current_input_name, minimum, maximum)

    def _updateLimit(self, axis_name: str | None, minimum: Any, maximum: Any) -> None:
        if axis_name == PlotContext.COUNT_AXIS:
            self._limits.count_limits = minimum, maximum
        elif axis_name == PlotContext.DENSITY_AXIS:
            self._limits.density_limits = minimum, maximum
        elif axis_name == PlotContext.DATE_AXIS:
            self._limits.date_limits = minimum, maximum
        elif axis_name == PlotContext.INDEX_AXIS:
            self._limits.index_limits = minimum, maximum
        elif axis_name == PlotContext.VALUE_AXIS:
            self._limits.value_limits = minimum, maximum

    def switchInputOnX(self, axis_type: str | None) -> None:
        self._x_current_input_name = axis_type
        self._updateWidgets()
        self._x_minimum_stack.switchToInput(axis_type)
        self._x_maximum_stack.switchToInput(axis_type)

    def switchInputOnY(self, axis_type: str | None) -> None:
        self._y_current_input_name = axis_type
        self._updateWidgets()
        self._y_minimum_stack.switchToInput(axis_type)
        self._y_maximum_stack.switchToInput(axis_type)


class LimitsCustomizationView(CustomizationView):
    def __init__(self) -> None:
        CustomizationView.__init__(self)

        limits_widget = LimitsWidget()
        self._limits_widget = limits_widget

        self.addHeading("X-axis")
        self.addRow("Minimum", limits_widget.x_minimum_stack)
        self.addRow("Maximum", limits_widget.x_maximum_stack)

        self.addHeading("Y-axis")
        self.addRow("Minimum", limits_widget.y_minimum_stack)
        self.addRow("Maximum", limits_widget.y_maximum_stack)

    def setAxisTypes(self, x_axis_type: str | None, y_axis_type: str | None) -> None:
        self._limits_widget.switchInputOnX(x_axis_type)
        self._limits_widget.switchInputOnY(y_axis_type)

    def revertCustomization(self, plot_config: PlotConfig) -> None:
        self._limits_widget.limits = plot_config.limits

    def applyCustomization(self, plot_config: PlotConfig) -> None:
        plot_config.limits = self._limits_widget.limits
