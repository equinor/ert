from typing import List

from ert.gui.ertwidgets.models.valuemodel import ValueModel
from ert_shared.libres_facade import LibresFacade
from res.config.active_range import ActiveRange
from res.config.rangestring import mask_to_rangestring


class ActiveRealizationsModel(ValueModel):
    def __init__(self, facade: LibresFacade):
        self.facade = facade
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
        size = self.facade.get_ensemble_size()
        return f"0-{size-1:d}"

    def getActiveRealizationsMask(self) -> List[bool]:
        return ActiveRange(
            rangestring=self.getValue(), length=self.facade.get_ensemble_size()
        ).mask
