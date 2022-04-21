from typing import List

from ert.ensemble_evaluator.activerange import ActiveRange, mask_to_rangestring
from ert_gui.ertwidgets.models.valuemodel import ValueModel
from ert_shared.libres_facade import LibresFacade


class ActiveRealizationsModel(ValueModel):
    def __init__(self, facade: LibresFacade):
        self.facade = facade
        ValueModel.__init__(self, self.getDefaultValue())
        self._custom = False

    def setValue(self, active_realizations):
        if (
            active_realizations is None
            or active_realizations.strip() == ""
            or active_realizations == self.getDefaultValue()
        ):
            self._custom = False
            ValueModel.setValue(self, self.getDefaultValue())
        else:
            self._custom = True
            ValueModel.setValue(self, active_realizations)

    def setValueFromMask(self, mask):
        self.setValue(mask_to_rangestring(mask))

    def getDefaultValue(self):
        size = self.facade.get_ensemble_size()
        return "0-%d" % (size - 1)

    def getActiveRealizationsMask(self) -> List[bool]:
        return ActiveRange(
            rangestring=self.getValue(), length=self.facade.get_ensemble_size()
        ).mask
