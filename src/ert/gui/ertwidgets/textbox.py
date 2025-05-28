from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QTextEdit

from .validationsupport import ValidationSupport

if TYPE_CHECKING:
    from ert.validation import StringDefinition

    from .models import TextModel


class TextBox(QTextEdit):
    """TextBox shows a multi line string. The data structure expected and sent to the
    getter and setter is a multi line string."""

    def __init__(
        self,
        model: TextModel,
        default_string: str = "",
        placeholder_text: str = "",
        minimum_width: int = 250,
    ) -> None:
        QTextEdit.__init__(self)
        self.setMinimumWidth(minimum_width)
        self._validation = ValidationSupport(self)
        self._validator: StringDefinition | None = None
        self._model = model
        self._enable_validation = True

        if placeholder_text:
            self.setPlaceholderText(placeholder_text)

        self.textChanged.connect(self.textBoxChanged)
        self.textChanged.connect(self.validateString)

        self._valid_color = self.palette().color(self.backgroundRole())
        self.setText(default_string)

        self._model.valueChanged.connect(self.modelChanged)
        self.modelChanged()

    def validateString(self) -> None:
        if self._enable_validation:
            string_to_validate = self.get_text
            if self._validator is not None:
                status = self._validator.validate(string_to_validate)

                palette = QPalette()
                if not status:
                    palette.setColor(
                        self.backgroundRole(), ValidationSupport.ERROR_COLOR
                    )
                    self.setPalette(palette)
                    self._validation.setValidationMessage(
                        str(status), ValidationSupport.EXCLAMATION
                    )
                else:
                    palette.setColor(self.backgroundRole(), self._valid_color)
                    self.setPalette(palette)
                    self._validation.setValidationMessage("")

    def emitChange(self, q_string: Any) -> None:
        self.textChanged.emit(str(q_string))

    def textBoxChanged(self) -> None:
        """Called whenever the contents of the textbox changes."""
        text: str | None = self.toPlainText()
        if not text:
            text = None

        self._model.setValue(text)

    def modelChanged(self) -> None:
        """Retrieves data from the model and inserts it into the textbox"""
        text = self._model.getValue()
        if text is None:
            text = ""
        # If model and view has same text, return
        if text == self.toPlainText():
            return
        self.setText(str(text))

    @property
    def model(self) -> TextModel:
        return self._model

    def setValidator(self, validator: StringDefinition) -> None:
        self._validator = validator

    def getValidationSupport(self) -> ValidationSupport:
        return self._validation

    def isValid(self) -> bool:
        return self._validation.isValid()

    @property
    def get_text(self) -> str:
        return self.toPlainText() or self.placeholderText()

    def enable_validation(self, enabled: bool) -> None:
        self._enable_validation = enabled

    def refresh(self) -> None:
        self.validateString()
