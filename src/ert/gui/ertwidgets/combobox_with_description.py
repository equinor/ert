from typing import Any, override

from PyQt6.QtCore import QEvent, QModelIndex, QObject, QPoint, QSignalBlocker, QSize, Qt
from PyQt6.QtGui import QColor, QMouseEvent, QRegion, QStandardItem, QStandardItemModel
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


class _ComboBoxGroupWidget(QLabel):
    def __init__(self, title: str) -> None:
        super().__init__(title)
        self.setStyleSheet(
            """
            padding: 5px 5px 5px 5px;
            background: rgba(0,0,0,0);
            font-style: italic;
            font-size: 14px;
            """
        )


class _ComboBoxItemWidget(QWidget):
    def __init__(
        self,
        label: str,
        description: str,
        *,
        enabled: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setSpacing(5)
        self.setStyleSheet("background: rgba(0,0,0,1);")
        self.label = QLabel(label)
        color = "color: rgba(192,192,192,80);" if not enabled else ";"
        self.label.setStyleSheet(
            f"""
            {color}
            padding-top: 5px;
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

        if (
            not group
            and is_enabled
            and (
                option.state & QStyle.StateFlag.State_Selected
                or option.state & QStyle.StateFlag.State_MouseOver
            )
        ):
            color = COLOR_HIGHLIGHT_LIGHT
            if option.palette.text().color().value() > 150:
                color = COLOR_HIGHLIGHT_DARK
            painter.fillRect(option.rect, color)

        widget = (
            _ComboBoxGroupWidget(group)
            if group
            else _ComboBoxItemWidget(label, description, enabled=bool(is_enabled))
        )
        widget.setStyle(option.widget.style())
        widget.resize(option.rect.size())

        painter.translate(option.rect.topLeft())
        widget.render(painter, QPoint(), QRegion(), QWidget.RenderFlag.DrawChildren)
        painter.restore()

    @override
    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        label = index.data(LABEL_ROLE)
        description = index.data(DESCRIPTION_ROLE)
        group = index.data(GROUP_TITLE_ROLE)
        widget = (
            _ComboBoxGroupWidget(group)
            if group
            else _ComboBoxItemWidget(label, description)
        )
        return widget.sizeHint()


class QComboBoxWithDescription(QComboBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setItemDelegate(_ComboBoxWithDescriptionDelegate(self))
        view = self.view()
        assert view is not None
        viewport = view.viewport()
        assert viewport is not None
        viewport.installEventFilter(self)

    @override
    def eventFilter(self, obj: QObject | None, event: QEvent | None) -> bool:
        view = self.view()
        assert view is not None
        if (
            obj is view.viewport()
            and event is not None
            and event.type()
            in {
                QEvent.Type.MouseButtonPress,
                QEvent.Type.MouseButtonRelease,
                QEvent.Type.MouseButtonDblClick,
            }
        ):
            assert isinstance(event, QMouseEvent)
            index = view.indexAt(event.position().toPoint())
            if index.data(GROUP_TITLE_ROLE):
                # Qt otherwise closes the popup even when the row is disabled.
                return True
        return super().eventFilter(obj, event)

    def addDescriptionItem(
        self, label: str, description: Any, group: str | None = None
    ) -> int:
        model = self.model()
        assert isinstance(model, QStandardItemModel)
        row = self.count()
        if group:
            group_row = self.findData(group, GROUP_TITLE_ROLE)
            if group_row == -1:
                header = QStandardItem(group)
                header.setData(group, GROUP_TITLE_ROLE)
                header.setFlags(Qt.ItemFlag.NoItemFlags)
                with QSignalBlocker(self):
                    model.appendRow(header)
                    if self.currentData(GROUP_TITLE_ROLE):
                        self.setCurrentIndex(-1)
                row = self.count()
            else:
                row = group_row + 1
                while row < self.count() and not self.itemData(row, GROUP_TITLE_ROLE):
                    row += 1

        super().insertItem(row, label)
        index = model.index(row, 0)
        model.setData(index, label, LABEL_ROLE)
        model.setData(index, description, DESCRIPTION_ROLE)
        if self.currentIndex() == -1:
            self.setCurrentIndex(row)
        return row

    @override
    def sizeHint(self) -> QSize:
        original_size_hint = super().sizeHint()
        new_width = int(original_size_hint.width() + 220)
        new_height = int(super().sizeHint().height() * 1.5)
        return QSize(new_width, new_height)
