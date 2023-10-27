from typing import List

from ert.gui.ertwidgets.models.valuemodel import ValueModel
from ert.validation import ActiveRange, mask_to_rangestring


class ActiveRealizationsModel(ValueModel):
    def __init__(self, ensemble_size: int):
        self.ensemble_size = ensemble_size
        ValueModel.__init__(self, self.getDefaultValue())
        self._custom = False

    def setValue(self, value: str):
        if value is None or value.strip() == "" or value == self.getDefaultValue():
            self._custom = False
            ValueModel.setValue(self, self.getDefaultValue())
        else:
            self._custom = True
            ValueModel.setValue(self, value)

    def setValueFromMask(self, mask):
        self.setValue(mask_to_rangestring(mask))

    def getDefaultValue(self):
        size = self.ensemble_size
        return f"0-{size-1:d}"

    def getActiveRealizationsMask(self) -> List[bool]:
        return ActiveRange(rangestring=self.getValue(), length=self.ensemble_size).mask
