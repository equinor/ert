from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets.models.valuemodel import ValueModel


class IterValueModel(ValueModel):
    def __init__(self, notifier: ErtNotifier, default_value=0):
        self._default_value = str(default_value)
        ValueModel.__init__(self, self.getDefaultValue())
        notifier.ertChanged.connect(self._caseChanged)

    def setValue(self, value):
        ValueModel.setValue(self, value)

    def getDefaultValue(self):
        return self._default_value

    def _caseChanged(self):
        ValueModel.setValue(self, self.getDefaultValue())
