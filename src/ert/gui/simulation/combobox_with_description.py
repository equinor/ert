from typing import Any

from PyQt6.QtCore import QModelIndex, QPoint, QSize
from PyQt6.QtGui import QColor, QPalette, QRegion
from PyQt6.QtWidgets import (
    QComboBox,
    QLabel,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

LABEL_ROLE = -3994
DESCRIPTION_ROLE = -4893
GROUP_TITLE_ROLE = -4894

COLOR_HIGHLIGHT_LIGHT = QColor(230, 230, 230, 255)
COLOR_HIGHLIGHT_DARK = QColor(60, 60, 60, 255)


class _ComboBoxItemWidget(QWidget):
    def __init__(
        self,
        label: str,
        description: str,
        enabled: bool = True,
        parent: QWidget | None = None,
        group: str | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setSpacing(5)
        # self.setPalette(parent.palette())
        self.label = QLabel(label, self)
        pd_top = "0px" if group else "5px"
        if group:
            self.group = QLabel(group, self)
            self.group.setStyleSheet(
                """
                padding-top: 5px;
                padding-left: 2px;
                font-style: italic;
                font-size: 14px;
            """
            )
            layout.addWidget(self.group)

        self.label.setStyleSheet(
            f"""
            padding-top:{pd_top};
            padding-left: 10px;
            font-weight: bold;
            font-size: 13px;
        """
        )
        self.description = QLabel(description, self)
        self.description.setStyleSheet(
            """
            padding-bottom: 10px;
            padding-left: 15px;
            font-style: italic;
            font-size: 12px;
        """
        )
        layout.addWidget(self.label)
        layout.addWidget(self.description)
        layout.setContentsMargins(0, 0, 0, 1)
        self.setLayout(layout)

    def set_palette(self):
        palette = QPalette()
        for group in [QPalette.ColorGroup.All]:
            palette.setColor(group, QPalette.ColorRole.Window, QColor("white"))
            palette.setColor(group, QPalette.ColorRole.WindowText, QColor("black"))
            palette.setColor(group, QPalette.ColorRole.Button, QColor("white"))
            palette.setColor(group, QPalette.ColorRole.ButtonText, QColor("black"))
            palette.setColor(group, QPalette.ColorRole.Base, QColor("white"))
            palette.setColor(group, QPalette.ColorRole.Text, QColor("black"))
            palette.setColor(group, QPalette.ColorRole.Highlight, QColor("pink"))
            palette.setColor(
                group, QPalette.ColorRole.HighlightedText, QColor("white")
            )  # The Qt default
        self.setPalette(palette)


class _ComboBoxWithDescriptionDelegate(QStyledItemDelegate):
    def __init__(self, parent: Any) -> None:
        super().__init__(parent)

    def paint(self, painter: Any, option: Any, index: Any) -> None:
        painter.save()

        label = index.data(LABEL_ROLE)
        description = index.data(DESCRIPTION_ROLE)
        group = index.data(GROUP_TITLE_ROLE)

        is_enabled = option.state & QStyle.StateFlag.State_Enabled

        if is_enabled and (
            option.state & QStyle.StateFlag.State_Selected
            or option.state & QStyle.StateFlag.State_MouseOver
        ):
            color = COLOR_HIGHLIGHT_LIGHT
            if option.palette.text().color().value() > 150:
                color = COLOR_HIGHLIGHT_DARK
            painter.fillRect(option.rect, color)

        widget = _ComboBoxItemWidget(
            label, description, is_enabled, group=group, parent=self.parent()
        )
        widget.setStyle(option.widget.style())
        widget.resize(option.rect.size())

        painter.translate(option.rect.topLeft())
        widget.render(painter, QPoint(), QRegion(), QWidget.RenderFlag.DrawChildren)
        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        label = index.data(LABEL_ROLE)
        description = index.data(DESCRIPTION_ROLE)
        group = index.data(GROUP_TITLE_ROLE)
        adjustment = QSize(0, 20) if group else QSize(0, 0)

        widget = _ComboBoxItemWidget(
            label, description, group=group, parent=self.parent()
        )
        return widget.sizeHint() + adjustment


class QComboBoxWithDescription(QComboBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setPalette(parent.palette())
        self.setItemDelegate(_ComboBoxWithDescriptionDelegate(parent))

    def addDescriptionItem(
        self, label: str, description: Any, group: str | None = None
    ) -> None:
        super().addItem(label)
        model = self.model()
        assert model is not None
        index = model.index(self.count() - 1, 0)
        model.setData(index, label, LABEL_ROLE)
        model.setData(index, description, DESCRIPTION_ROLE)
        model.setData(index, group, GROUP_TITLE_ROLE)

    def sizeHint(self) -> QSize:
        original_size_hint = super().sizeHint()
        new_width = int(original_size_hint.width() + 220)
        new_height = int(super().sizeHint().height() * 1.5)
        return QSize(new_width, new_height)
