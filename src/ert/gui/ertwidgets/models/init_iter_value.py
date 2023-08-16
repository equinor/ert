from ert.gui.ertwidgets.models.valuemodel import ValueModel


class IterValueModel(ValueModel):
    def __init__(self, default_value=0):
        self._default_value = str(default_value)
        ValueModel.__init__(self, self.getDefaultValue())

    def setValue(self, value):
        ValueModel.setValue(self, value)

    def getDefaultValue(self):
        return self._default_value
