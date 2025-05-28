from collections.abc import Sequence
from uuid import UUID

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon, QKeyEvent
from PyQt6.QtWidgets import (
    QCompleter,
    QHBoxLayout,
    QInputDialog,
    QLineEdit,
    QMessageBox,
    QToolButton,
    QWidget,
)

from .validationsupport import ValidationSupport


class AutoCompleteLineEdit(QLineEdit):
    # http://blog.elentok.com/2011/08/autocomplete-textbox-for-multiple.html
    def __init__(self, items: Sequence[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._separators = [",", " "]

        self._completer = QCompleter(items, self)
        self._completer.setWidget(self)
        self._completer.activated.connect(self.__insertCompletion)
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

        self.__keysToIgnore = [
            Qt.Key.Key_Enter,
            Qt.Key.Key_Return,
            Qt.Key.Key_Escape,
            Qt.Key.Key_Tab,
        ]

    def __insertCompletion(self, completion: str) -> None:
        extra = len(completion) - len(self._completer.completionPrefix())
        extra_text = completion[-extra:]
        extra_text += ", "
        self.setText(self.text() + extra_text)

    def textUnderCursor(self) -> str:
        text = self.text()
        text_under_cursor = ""
        i = self.cursorPosition() - 1
        while i >= 0 and text[i] not in self._separators:
            text_under_cursor = text[i] + text_under_cursor
            i -= 1
        return text_under_cursor

    def keyPressEvent(self, a0: QKeyEvent | None) -> None:
        popup = self._completer.popup()
        if (
            popup is not None
            and popup.isVisible()
            and a0 is not None
            and a0.key() in self.__keysToIgnore
        ):
            a0.ignore()
            return

        super().keyPressEvent(a0)

        completion_prefix = self.textUnderCursor()
        if completion_prefix != self._completer.completionPrefix():
            self.__updateCompleterPopupItems(completion_prefix)
        if a0 is not None and len(a0.text()) > 0 and len(completion_prefix) > 0:
            self._completer.complete()
        if popup is not None and len(completion_prefix) == 0:
            popup.hide()

    def __updateCompleterPopupItems(self, completionPrefix: str) -> None:
        self._completer.setCompletionPrefix(completionPrefix)
        popup = self._completer.popup()
        assert popup is not None
        model = self._completer.completionModel()
        assert model is not None
        popup.setCurrentIndex(model.index(0, 0))


class ListEditBox(QWidget):
    ITEM_DOES_NOT_EXIST_MSG = "The item: '%s' is not a possible choice."
    NO_ITEMS_SPECIFIED_MSG = "The list must contain at least one item or * (for all)."
    DEFAULT_MSG = "A list of comma separated ensemble names or * for all."

    def __init__(self, possible_items: dict[UUID, str]) -> None:
        QWidget.__init__(self)

        self._editing = True
        self._possible_items_dict = possible_items
        self._possible_items = list(possible_items.values())

        self._list_edit_line = AutoCompleteLineEdit(self._possible_items, self)
        self._list_edit_line.setMinimumWidth(350)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._list_edit_line)

        dialog_button = QToolButton(self)
        dialog_button.setIcon(QIcon("img:add_circle_outlined.svg"))
        dialog_button.setIconSize(QSize(16, 16))
        dialog_button.clicked.connect(self.addChoice)

        layout.addWidget(dialog_button)

        self.setLayout(layout)

        self._validation_support = ValidationSupport(self)
        self._valid_color = self._list_edit_line.palette().color(
            self._list_edit_line.backgroundRole()
        )

        self._list_edit_line.setText("")
        self._editing = False

        self._list_edit_line.editingFinished.connect(self.validateList)
        self._list_edit_line.textChanged.connect(self.validateList)

        self.validateList()

    def getListText(self) -> str:
        text = str(self._list_edit_line.text())
        text = "".join(text.split())
        return text

    def getItems(self) -> dict[UUID, str]:
        text = self.getListText()
        items = text.split(",")

        if len(items) == 1 and items[0] == "*":
            return self._possible_items_dict

        result = {}
        for item in items:
            item = item.strip()
            for uuid, name in self._possible_items_dict.items():
                if name == item:
                    result[uuid] = name
                    break

        return result

    def validateList(self) -> None:
        """Called whenever the list is modified"""
        palette = self._list_edit_line.palette()
        items = self.getItems()
        valid = True
        message = ""

        if len(items) == 0:
            valid = False
            message = ListEditBox.NO_ITEMS_SPECIFIED_MSG
        else:
            for _, name in items.items():
                if name not in self._possible_items_dict.values():
                    valid = False
                    message = ListEditBox.ITEM_DOES_NOT_EXIST_MSG % name
                    break

        validity_type = ValidationSupport.WARNING
        color = ValidationSupport.ERROR_COLOR if not valid else self._valid_color
        self._validation_support.setValidationMessage(message, validity_type)
        self._list_edit_line.setToolTip(message)
        palette.setColor(self._list_edit_line.backgroundRole(), color)
        self._list_edit_line.setPalette(palette)

        if valid:
            self._list_edit_line.setToolTip(ListEditBox.DEFAULT_MSG)

    def addChoice(self) -> None:
        if len(self._possible_items) == 0:
            QMessageBox.information(
                self, "No items", "No items available for selection!"
            )
        else:
            item, ok = QInputDialog.getItem(
                self,
                "Select an ensemble",
                "Select an ensemble to add to the ensemble list:",
                self._possible_items,
            )

            if ok:
                item = str(item).strip()
                text = str(self._list_edit_line.text()).rstrip()

                if len(text) == 0:
                    text = item + ", "
                elif text.endswith(","):
                    text += " " + item
                else:
                    text += ", " + item

                self._list_edit_line.setText(text)

    def getValidationSupport(self) -> ValidationSupport:
        return self._validation_support

    def isValid(self) -> bool:
        return self._validation_support.isValid()
