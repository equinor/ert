from typing import Any, List, Optional, Tuple

from qtpy.QtCore import QModelIndex, QPoint, QSize
from qtpy.QtGui import QColor, QRegion
from qtpy.QtWidgets import (
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

COLOR_HIGHLIGHT_LIGHT = QColor(230, 230, 230, 255)
COLOR_HIGHLIGHT_DARK = QColor(60, 60, 60, 255)


class ComboBoxItemWidget(QWidget):
    def __init__(
        self, label: str, description: str, parent: Optional[QWidget] = None
    ) -> None:
        super(ComboBoxItemWidget, self).__init__(parent)
        layout = QVBoxLayout()
        layout.setSpacing(2)
        self.setStyleSheet("background: rgba(0,0,0,1);")
        self.label = QLabel(label)
        self.label.setStyleSheet(
            """
            padding-top:5px;
            padding-left: 5px;
            background: rgba(0,0,0,0);
            font-weight: bold;
            font-size: 13px;
        """
        )
        self.description = QLabel(description)
        self.description.setStyleSheet(
            """
            padding-bottom: 10px;
            padding-left: 10px;
            background: rgba(0,0,0,0);
            font-style: italic;
            font-size: 12px;
        """
        )
        layout.addWidget(self.label)
        layout.addWidget(self.description)
        layout.setContentsMargins(0, 0, 0, 1)
        self.setLayout(layout)


class ComboBoxWithDescriptionDelegate(QStyledItemDelegate):
    def paint(self, painter: Any, option: Any, index: Any) -> None:
        painter.save()

        label = index.data(LABEL_ROLE)
        description = index.data(DESCRIPTION_ROLE)

        if (
            option.state & QStyle.State_Selected  # type: ignore
            or option.state & QStyle.State_MouseOver  # type: ignore
        ):
            color = COLOR_HIGHLIGHT_LIGHT
            if option.palette.text().color().value() > 150:
                color = COLOR_HIGHLIGHT_DARK
            painter.fillRect(option.rect, color)

        widget = ComboBoxItemWidget(label, description)
        widget.setStyle(option.widget.style())
        widget.resize(option.rect.size())

        painter.translate(option.rect.topLeft())
        widget.render(painter, QPoint(), QRegion(), QWidget.RenderFlag.DrawChildren)
        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        label = index.data(LABEL_ROLE)
        description = index.data(DESCRIPTION_ROLE)

        widget = ComboBoxItemWidget(label, description)
        return widget.sizeHint()


class QComboBoxWithDescription(QComboBox):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super(QComboBoxWithDescription, self).__init__(parent)
        self.setItemDelegate(ComboBoxWithDescriptionDelegate(self))

    def addItem(self, label: Optional[str], description: Any):  # type: ignore
        super(QComboBoxWithDescription, self).addItem(label)
        model = self.model()
        assert model is not None
        index = model.index(self.count() - 1, 0)
        model.setData(index, label, LABEL_ROLE)
        model.setData(index, description, DESCRIPTION_ROLE)

    def addItems(self, items: List[Tuple[str, Any]]):  # type: ignore
        for label, description in items:
            self.addItem(label, description)
