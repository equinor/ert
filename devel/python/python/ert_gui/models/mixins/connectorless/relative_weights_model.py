from ert_gui.models import ErtConnector
from ert.enkf import ModelConfig
from ert_gui.models.mixins import BasicModelMixin

class RelativeWeightsModel(ErtConnector, BasicModelMixin):
    def getValue(self):
        """ @rtype: list """
        ws = self.getModelConfig().getWeightsAsString()
        return map(int, ws)

    def setValue(self, weights):
        self.getModelConfig().setWeights(str(weights))
        self.observable().notify(self.VALUE_CHANGED_EVENT)

    def getModelConfig(self):
        """ @rtype: ModelConfig """
        return self.ert().getModelConfig()
