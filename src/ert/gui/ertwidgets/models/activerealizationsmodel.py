from collections.abc import Collection

from ert.gui.ertwidgets.models.valuemodel import ValueModel
from ert.validation import ActiveRange, mask_to_rangestring


class ActiveRealizationsModel(ValueModel):
    def __init__(self, ensemble_size: int, show_default: bool = True) -> None:
        self.default_value = f"0-{ensemble_size - 1:d}" if show_default else None
        self.ensemble_size = ensemble_size
        ValueModel.__init__(self, self.default_value)

    def setValue(self, value: str | None) -> None:
        if not value or not value.strip() or value == self.default_value:
            ValueModel.setValue(self, self.default_value)
        else:
            ValueModel.setValue(self, value)

    def setValueFromMask(self, mask: Collection[bool | int]) -> None:
        self.setValue(mask_to_rangestring(mask))

    def getActiveRealizationsMask(self) -> list[bool]:
        return ActiveRange(rangestring=self.getValue(), length=self.ensemble_size).mask
