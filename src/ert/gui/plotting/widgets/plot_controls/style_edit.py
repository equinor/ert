from collections.abc import Callable

from PyQt6.QtCore import QSignalBlocker
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from ert.gui.plotting.customization_dialog.style_chooser import StyleChooser
from ert.gui.plotting.utils import PlotStyle

STYLE_NAME_WIDTH = 52


class _StyleEdit(QWidget):
    def __init__(
        self,
        connection_point: Callable[..., object],
        initial_style: PlotStyle,
        title: str,
        object_name: str,
        line_style_set: str,
        display_name: str | None = None,
        tool_tip: str | None = None,
    ) -> None:
        super().__init__()
        self._connection_point = connection_point
        self._title = title
        self._initial_style = PlotStyle(title)
        self._initial_style.copy_style_from(initial_style)
        self._style = PlotStyle(title)
        self._style.copy_style_from(initial_style)
        self.setObjectName(object_name)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        style_name = QLabel(display_name or title.removesuffix(" style"))
        style_name.setFixedWidth(STYLE_NAME_WIDTH)
        if tool_tip is not None:
            style_name.setToolTip(tool_tip)
        layout.addWidget(style_name)

        self._style_chooser = StyleChooser(line_style_set=line_style_set, compact=True)
        self._style_chooser.setStyle(self._style)
        self._style_chooser.styleChanged.connect(self._update_style)
        layout.addWidget(self._style_chooser)

    def _update_style(self) -> None:
        self._style = self._style_chooser.get_style()
        self._connection_point()

    def get_style(self) -> PlotStyle:
        style = PlotStyle(self._title)
        style.copy_style_from(self._style)
        return style

    def set_style(self, style: PlotStyle) -> None:
        self._style.copy_style_from(style)
        with QSignalBlocker(self._style_chooser):
            self._style_chooser.setStyle(self._style)

    def reset_style(self) -> None:
        self.set_style(self._initial_style)
