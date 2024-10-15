from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from qtpy.QtGui import QPalette
from qtpy.QtWidgets import QTextEdit

from .validationsupport import ValidationSupport

if TYPE_CHECKING:
    from ert.validation import ArgumentDefinition

    from .models import TextModel


class MultiLineStringBox(QTextEdit):
    """MultiLineStringBox shows a multiline string. The data structure expected and sent to the
    getter and setter is a multiline string."""

    def __init__(
        self,
        model: TextModel,
        default_string: str = "",
        placeholder_text: str = "",
        minimum_width: int = 250,
        readonly: bool = False,
    ):
        QTextEdit.__init__(self)
        self.setMinimumWidth(minimum_width)
        self._validation = ValidationSupport(self)
        self._validator: Optional[ArgumentDefinition] = None
        self._model = model
        self._enable_validation = True

        if placeholder_text:
            self.setPlaceholderText(placeholder_text)
        self.textChanged.connect(self.stringBoxChanged)

        self.textChanged.connect(self.validateString)

        self._valid_color = self.palette().color(self.backgroundRole())
        self.setText(default_string)

        self._model.valueChanged.connect(self.modelChanged)
        self.modelChanged()
        self.setReadOnly(readonly)

    def validateString(self) -> None:
        if not self._enable_validation or self._validator is None:
            return

        string_to_validate = self.toPlainText()
        if not string_to_validate and self.placeholderText():
            string_to_validate = self.placeholderText()

        validation_success = self._validator.validate(string_to_validate)

        palette = self.palette()
        if not validation_success:
            palette.setColor(QPalette.ColorRole.Base, ValidationSupport.ERROR_COLOR)
            self.setPalette(palette)
            self._validation.setValidationMessage(
                str(validation_success), ValidationSupport.EXCLAMATION
            )
        else:
            palette.setColor(QPalette.ColorRole.Base, self._valid_color)
            self.setPalette(palette)
            self._validation.setValidationMessage("")

    def emitChange(self, q_string: Any) -> None:
        self.textChanged.emit(str(q_string))

    def stringBoxChanged(self) -> None:
        """Called whenever the contents of the textedit changes."""
        text: Optional[str] = self.get_text
        if not text:
            text = None

        self._model.setValue(text)

    def modelChanged(self) -> None:
        """Retrieves data from the model and inserts it into the textedit"""
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

    def setValidator(self, validator: ArgumentDefinition) -> None:
        self._validator = validator

    def getValidationSupport(self) -> ValidationSupport:
        return self._validation

    def isValid(self) -> bool:
        return self._validation.isValid()

    @property
    def get_text(self) -> str:
        return self.toPlainText() if self.toPlainText() else self.placeholderText()

    def enable_validation(self, enabled: bool) -> None:
        self._enable_validation = enabled

    def refresh(self) -> None:
        self.validateString()
