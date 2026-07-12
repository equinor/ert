from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QSpacerItem,
    QSpinBox,
    QWidget,
)

from ert.gui.plotting.widgets import ClearableLineEdit

from .style_chooser import STYLESET_DEFAULT, StyleChooser

if TYPE_CHECKING:
    from ert.gui.plotting.utils import PlotConfig


class CustomizationView(QWidget):
    def __init__(self) -> None:
        QWidget.__init__(self)

        self._layout = QFormLayout()
        self.setLayout(self._layout)
        self._widgets: dict[str, QWidget] = {}

    def add_row(self, title: str, widget: QWidget) -> None:
        self._layout.addRow(title, widget)

    def add_line_edit(
        self,
        attribute_name: str,
        title: str,
        tool_tip: str | None = None,
        placeholder: str = "",
    ) -> None:
        self[attribute_name] = ClearableLineEdit(placeholder=placeholder)
        self.add_row(title, self[attribute_name])

        if tool_tip is not None:
            self[attribute_name].setToolTip(tool_tip)

        def getter(self: Any) -> str | None:
            value: str | None = str(self[attribute_name].text())
            if not value:
                value = None
            return value

        def setter(self: Any, value: str | None) -> None:
            if value is None:
                value = ""
            self[attribute_name].setText(str(value))

        self.update_property(attribute_name, getter, setter)

    def add_check_box(
        self, attribute_name: str, title: str, tool_tip: str | None = None
    ) -> None:
        self[attribute_name] = QCheckBox()
        self.add_row(title, self[attribute_name])

        if tool_tip is not None:
            self[attribute_name].setToolTip(tool_tip)

        def getter(self: Any) -> bool:
            return self[attribute_name].isChecked()

        def setter(self: Any, value: bool) -> None:
            self[attribute_name].setChecked(value)

        self.update_property(attribute_name, getter, setter)

    def add_integer_selection_box(
        self,
        attribute_name: str,
        title: str,
        tool_tip: str | None = None,
        min_value: int = 1,
        max_value: int = 10,
        single_step: int = 1,
    ) -> QSpinBox:
        sb = QSpinBox()
        self[attribute_name] = sb
        sb.setMaximumHeight(25)
        sb_layout = QHBoxLayout()
        sb_layout.addWidget(sb)
        sb_layout.addStretch()
        self.add_row(title, sb_layout)  # type: ignore

        if tool_tip is not None:
            sb.setToolTip(tool_tip)

        sb.setMinimum(min_value)
        sb.setMaximum(max_value)
        sb.setSingleStep(single_step)

        def getter(self: Any) -> QWidget:
            return self[attribute_name].value()

        def setter(self: Any, value: QWidget) -> None:
            self[attribute_name].setValue(value)

        self.update_property(attribute_name, getter, setter)
        return sb

    def add_style_chooser(
        self,
        attribute_name: str,
        title: str,
        tool_tip: str | None = None,
        line_style_set: str = STYLESET_DEFAULT,
    ) -> None:
        style_chooser = StyleChooser(line_style_set=line_style_set)
        self[attribute_name] = style_chooser
        self.add_row(title, self[attribute_name])

        if tool_tip is not None:
            self[attribute_name].setToolTip(tool_tip)

        def getter(self: Any) -> QWidget:
            return self[attribute_name].get_style()

        def setter(self: Any, style: QWidget) -> None:
            self[attribute_name].setStyle(style)

        self.update_property(attribute_name, getter, setter)

    def update_property(
        self,
        attribute_name: str,
        getter: Callable[[Any], Any],
        setter: Callable[[Any, Any], None],
    ) -> None:
        setattr(self.__class__, attribute_name, property(getter, setter))

    def add_spacing(self, pixels: int = 10) -> None:
        self._layout.addItem(QSpacerItem(1, pixels))

    def add_heading(self, title: str) -> None:
        self.add_spacing(10)
        self._layout.addRow(title, None)
        self.add_spacing(1)

    def __getitem__(self, item: str) -> QWidget:
        return self._widgets[item]

    def __setitem__(self, key: str, value: QWidget) -> None:
        self._widgets[key] = value

    def apply_customization(self, plot_config: PlotConfig) -> None:
        raise NotImplementedError(
            f"Class '{self.__class__.__name__}' has not implemented "
            "the apply_customization() function!"
        )

    def revert_customization(self, plot_config: PlotConfig) -> None:
        raise NotImplementedError(
            f"Class '{self.__class__.__name__}' has not implemented "
            "the revert_customization() function!"
        )


class WidgetProperty:
    def __get__(self, instance: Any, owner: Any) -> Any:
        raise UserWarning("Property is invalid!")

    def __set__(self, instance: Any, value: Any) -> Any:
        raise UserWarning("Property is invalid!")
