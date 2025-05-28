from collections.abc import Collection

from ert.gui.ertwidgets.models.valuemodel import ValueModel
from ert.validation import ActiveRange, mask_to_rangestring


class ActiveRealizationsModel(ValueModel):
    def __init__(self, ensemble_size: int, show_default: bool = True) -> None:
        self.show_default = show_default
        self.ensemble_size = ensemble_size
        ValueModel.__init__(self, self.getDefaultValue())
        self._custom = False

    def setValue(self, value: str | None) -> None:
        if not value or not value.strip() or value == self.getDefaultValue():
            self._custom = False
            ValueModel.setValue(self, self.getDefaultValue())
        else:
            self._custom = True
            ValueModel.setValue(self, value)

    def setValueFromMask(self, mask: Collection[bool | int]) -> None:
        self.setValue(mask_to_rangestring(mask))

    def getDefaultValue(self) -> str | None:
        if self.show_default:
            size = self.ensemble_size
            return f"0-{size - 1:d}"
        return None

    def getActiveRealizationsMask(self) -> list[bool]:
        return ActiveRange(rangestring=self.getValue(), length=self.ensemble_size).mask
