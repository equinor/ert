from typing import Optional

from ert.gui.ertwidgets.models.valuemodel import ValueModel


class TextModel(ValueModel):
    def __init__(
        self,
        default_value: str,
    ) -> None:
        self.default_value = default_value
        super().__init__(self.getDefaultValue())

    def setValue(self, value: Optional[str]) -> None:
        if not value or not value.strip() or value == self.getDefaultValue():
            ValueModel.setValue(self, self.getDefaultValue())
        else:
            ValueModel.setValue(self, value)

    def getDefaultValue(self) -> str:
        return self.default_value
