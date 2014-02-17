from ert.enkf.enums.enkf_obs_impl_type_enum import EnkfObservationImplementationType
from ert.enkf.enums.enkf_var_type_enum import EnkfVarType
from ert_gui.models import ErtConnector

class ErtSummary(ErtConnector):

    def getForwardModels(self):
        """ @rtype: list of str """
        forward_model  = self.ert().getModelConfig().getForwardModel()
        return [job for job in forward_model.joblist()]

    def getParameters(self):
        """ @rtype: list of str """
        parameters = self.ert().ensembleConfig().getKeylistFromVarType(EnkfVarType.PARAMETER)
        return sorted([parameter for parameter in parameters], key=lambda k : k.lower())


    def getObservations(self):
        """ @rtype: list of str """
        gen_obs = self.ert().getObservations().getTypedKeylist(EnkfObservationImplementationType.GEN_OBS)
        summary_obs = self.ert().getObservations().getTypedKeylist(EnkfObservationImplementationType.SUMMARY_OBS)
        obs_keys = [observation for observation in gen_obs] + [summary for summary in summary_obs]
        return sorted(obs_keys, key=lambda k : k.lower())






