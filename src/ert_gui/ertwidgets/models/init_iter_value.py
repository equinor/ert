from ert_gui.tools.load_results import LoadResultsModel
from ert_gui.ertwidgets.models.valuemodel import ValueModel
from ert_shared import ERT


class IterValueModel(ValueModel):
    def __init__(self):
        ValueModel.__init__(self, self.getDefaultValue())
        ERT.ertChanged.connect(self._caseChanged)

    def setValue(self, iter_value):
        ValueModel.setValue(self, iter_value)

    def getDefaultValue(self):
        return "0"

    def _caseChanged(self):
        ValueModel.setValue(self, self.getDefaultValue())
