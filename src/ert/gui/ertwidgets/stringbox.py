from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from qtpy.QtGui import QPalette
from qtpy.QtWidgets import QLineEdit

from ert.gui.ertwidgets import ValidationSupport

if TYPE_CHECKING:
    from ert.validation import ArgumentDefinition

    from .models import TextModel


class StringBox(QLineEdit):
    """StringBox shows a string. The data structure expected and sent to the
    getter and setter is a string."""

    def __init__(
        self,
        model: TextModel,
        default_string: str = "",
        continuous_update: bool = False,
        placeholder_text: str = "",
        minimum_width: int = 250,
    ):
        """
        :type model: ert.gui.ertwidgets.models.valuemodel.ValueModel
        :type help_link: str
        :type default_string: str
        :type continuous_update: bool
        """
        QLineEdit.__init__(self)
        self.setMinimumWidth(minimum_width)
        self._validation = ValidationSupport(self)
        self._validator: Optional[ArgumentDefinition] = None
        self._model = model
        if placeholder_text:
            self.setPlaceholderText(placeholder_text)
        self.editingFinished.connect(self.stringBoxChanged)
        self.editingFinished.connect(self.validateString)

        if continuous_update:
            self.textChanged.connect(self.stringBoxChanged)

        self.textChanged.connect(self.validateString)

        self._valid_color = self.palette().color(self.backgroundRole())
        self.setText(default_string)

        self._model.valueChanged.connect(self.modelChanged)
        self.modelChanged()

    def validateString(self) -> None:
        string_to_validate = str(self.text())
        if not string_to_validate and self.placeholderText():
            string_to_validate = self.placeholderText()
        if self._validator is not None:
            status = self._validator.validate(string_to_validate)

            palette = QPalette()
            if not status:
                palette.setColor(self.backgroundRole(), ValidationSupport.ERROR_COLOR)
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

    def stringBoxChanged(self) -> None:
        """Called whenever the contents of the editline changes."""
        text: Optional[str] = self.text()
        if not text:
            text = None

        self._model.setValue(text)

    def modelChanged(self) -> None:
        """Retrieves data from the model and inserts it into the edit line"""
        text = self._model.getValue()
        if text is None:
            text = ""
        # If model and view has same text, return
        if text == self.text():
            return
        self.setText(str(text))

    @property
    def model(self) -> TextModel:
        return self._model

    def setValidator(self, validator: ArgumentDefinition) -> None:  # type: ignore
        self._validator = validator

    def getValidationSupport(self) -> ValidationSupport:
        return self._validation

    def isValid(self) -> bool:
        return self._validation.isValid()

    @property
    def get_text(self) -> str:
        return self.text() if self.text() else self.placeholderText()
