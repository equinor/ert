from typing import Any

from PyQt6.QtCore import QModelIndex, QPoint, QSize
from PyQt6.QtGui import QColor, QRegion
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
        self.setStyleSheet("background: rgba(0,0,0,1);")
        self.label = QLabel(label)
        color = "color: rgba(192,192,192,80);" if not enabled else ";"
        pd_top = "0px" if group else "5px"
        if group:
            self.group = QLabel(group)
            self.group.setStyleSheet(
                f"""
                {color}
                padding-top: 5px;
                padding-left: 2px;
                background: rgba(0,0,0,0);
                font-style: italic;
                font-size: 14px;
            """
            )
            layout.addWidget(self.group)

        self.label.setStyleSheet(
            f"""
            {color}
            padding-top:{pd_top};
            padding-left: 10px;
            background: rgba(0,0,0,0);
            font-weight: bold;
            font-size: 13px;
        """
        )
        self.description = QLabel(description)
        self.description.setStyleSheet(
            f"""
            {color}
            padding-bottom: 10px;
            padding-left: 15px;
            background: rgba(0,0,0,0);
            font-style: italic;
            font-size: 12px;
        """
        )
        layout.addWidget(self.label)
        layout.addWidget(self.description)
        layout.setContentsMargins(0, 0, 0, 1)
        self.setLayout(layout)


class _ComboBoxWithDescriptionDelegate(QStyledItemDelegate):
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

        widget = _ComboBoxItemWidget(label, description, is_enabled, group=group)
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

        widget = _ComboBoxItemWidget(label, description, group)
        return widget.sizeHint() + adjustment


class QComboBoxWithDescription(QComboBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setItemDelegate(_ComboBoxWithDescriptionDelegate(self))

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
