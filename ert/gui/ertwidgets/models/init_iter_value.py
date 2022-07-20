from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets.models.valuemodel import ValueModel


class IterValueModel(ValueModel):
    def __init__(self, notifier: ErtNotifier):
        ValueModel.__init__(self, self.getDefaultValue())
        notifier.ertChanged.connect(self._caseChanged)

    def setValue(self, value):
        ValueModel.setValue(self, value)

    def getDefaultValue(self):
        return "0"

    def _caseChanged(self):
        ValueModel.setValue(self, self.getDefaultValue())
