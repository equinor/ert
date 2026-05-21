from __future__ import annotations

from copy import copy
from datetime import date
from typing import TYPE_CHECKING, Any, ClassVar, cast

from PyQt6.QtGui import QDoubleValidator, QIntValidator
from PyQt6.QtWidgets import QLabel, QLineEdit, QStackedWidget
from typing_extensions import override

from ert.gui.tools.plot.utils import PlotContext, PlotLimits
from ert.gui.tools.plot.widgets import ClearableLineEdit, CustomDateEdit

from .customization_view import CustomizationView

if TYPE_CHECKING:
    from ert.gui.tools.plot.utils import PlotConfig


class StackedInput(QStackedWidget):
    def __init__(self) -> None:
        QStackedWidget.__init__(self)
        self._inputs: dict[str | None, QLineEdit | QLabel | CustomDateEdit] = {}
        self._index_map: dict[str | None, int] = {}
        self.add_input(PlotContext.UNKNOWN_AXIS, QLabel("Fixed"))
        self._current_name: str | None = PlotContext.UNKNOWN_AXIS

    def add_input(
        self, name: str | None, widget: QLineEdit | QLabel | CustomDateEdit
    ) -> None:
        index = self.addWidget(widget)
        self._inputs[name] = widget
        self._index_map[name] = index

    def switch_to_input(self, name: str | None) -> None:
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
        self.add_input(
            PlotContext.COUNT_AXIS,
            self.create_integer_line_edit(minimum=0, placeholder="Default value"),
        )
        self.add_input(PlotContext.DATE_AXIS, CustomDateEdit())
        self.add_input(
            PlotContext.DENSITY_AXIS,
            self.create_double_line_edit(minimum=0, placeholder="Default value"),
        )
        self.add_input(
            PlotContext.INDEX_AXIS,
            self.create_integer_line_edit(minimum=0, placeholder="Default value"),
        )
        self.add_input(
            PlotContext.VALUE_AXIS,
            self.create_double_line_edit(placeholder="Default value"),
        )

    @staticmethod
    def create_double_line_edit(
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
    def create_integer_line_edit(
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

    def set_value(self, axis_name: str | None, value: Any) -> None:
        input_ = self._inputs[axis_name]

        if axis_name in LimitsStack.NUMBER_AXIS and (
            issubclass(type(input_), QLineEdit) or issubclass(type(input_), QLabel)
        ):
            input_ = cast(QLineEdit | QLabel, input_)
            input_.setText(str(value) if value is not None else "")
        elif axis_name == PlotContext.DATE_AXIS and type(input_) is CustomDateEdit:
            input_.setDate(value)

    def get_value(self, axis_name: str | None) -> float | int | date | None:
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
        self._x_minimum_stack.set_value(PlotContext.DATE_AXIS, limits.date_minimum)
        self._x_maximum_stack.set_value(PlotContext.DATE_AXIS, limits.date_maximum)
        self._y_minimum_stack.set_value(PlotContext.DATE_AXIS, limits.date_minimum)
        self._y_maximum_stack.set_value(PlotContext.DATE_AXIS, limits.date_maximum)

        self._x_minimum_stack.set_value(
            PlotContext.DENSITY_AXIS, limits.density_minimum
        )
        self._x_maximum_stack.set_value(
            PlotContext.DENSITY_AXIS, limits.density_maximum
        )
        self._y_minimum_stack.set_value(
            PlotContext.DENSITY_AXIS, limits.density_minimum
        )
        self._y_maximum_stack.set_value(
            PlotContext.DENSITY_AXIS, limits.density_maximum
        )

        self._x_minimum_stack.set_value(PlotContext.COUNT_AXIS, limits.count_minimum)
        self._x_maximum_stack.set_value(PlotContext.COUNT_AXIS, limits.count_maximum)
        self._y_minimum_stack.set_value(PlotContext.COUNT_AXIS, limits.count_minimum)
        self._y_maximum_stack.set_value(PlotContext.COUNT_AXIS, limits.count_maximum)

        self._x_minimum_stack.set_value(PlotContext.INDEX_AXIS, limits.index_minimum)
        self._x_maximum_stack.set_value(PlotContext.INDEX_AXIS, limits.index_maximum)
        self._y_minimum_stack.set_value(PlotContext.INDEX_AXIS, limits.index_minimum)
        self._y_maximum_stack.set_value(PlotContext.INDEX_AXIS, limits.index_maximum)

        self._x_minimum_stack.set_value(PlotContext.VALUE_AXIS, limits.value_minimum)
        self._x_maximum_stack.set_value(PlotContext.VALUE_AXIS, limits.value_maximum)
        self._y_minimum_stack.set_value(PlotContext.VALUE_AXIS, limits.value_minimum)
        self._y_maximum_stack.set_value(PlotContext.VALUE_AXIS, limits.value_maximum)

    def _updateLimits(self) -> None:
        if self._x_current_input_name is not PlotContext.UNKNOWN_AXIS:
            minimum = self._x_minimum_stack.get_value(self._x_current_input_name)
            maximum = self._x_maximum_stack.get_value(self._x_current_input_name)
            self._updateLimit(self._x_current_input_name, minimum, maximum)

        if self._y_current_input_name is not PlotContext.UNKNOWN_AXIS:
            minimum = self._y_minimum_stack.get_value(self._y_current_input_name)
            maximum = self._y_maximum_stack.get_value(self._y_current_input_name)
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

    def switch_input_on_x(self, axis_type: str | None) -> None:
        self._x_current_input_name = axis_type
        self._updateWidgets()
        self._x_minimum_stack.switch_to_input(axis_type)
        self._x_maximum_stack.switch_to_input(axis_type)

    def switch_input_on_y(self, axis_type: str | None) -> None:
        self._y_current_input_name = axis_type
        self._updateWidgets()
        self._y_minimum_stack.switch_to_input(axis_type)
        self._y_maximum_stack.switch_to_input(axis_type)


class LimitsCustomizationView(CustomizationView):
    def __init__(self) -> None:
        CustomizationView.__init__(self)

        limits_widget = LimitsWidget()
        self._limits_widget = limits_widget

        self.add_heading("X-axis")
        self.add_row("Minimum", limits_widget.x_minimum_stack)
        self.add_row("Maximum", limits_widget.x_maximum_stack)

        self.add_heading("Y-axis")
        self.add_row("Minimum", limits_widget.y_minimum_stack)
        self.add_row("Maximum", limits_widget.y_maximum_stack)

    def set_axis_types(self, x_axis_type: str | None, y_axis_type: str | None) -> None:
        self._limits_widget.switch_input_on_x(x_axis_type)
        self._limits_widget.switch_input_on_y(y_axis_type)

    @override
    def revert_customization(self, plot_config: PlotConfig) -> None:
        self._limits_widget.limits = plot_config.limits

    @override
    def apply_customization(self, plot_config: PlotConfig) -> None:
        plot_config.limits = self._limits_widget.limits
