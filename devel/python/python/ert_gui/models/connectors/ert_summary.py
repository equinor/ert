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

        keys = []
        for key in summary_obs:
            data_key = self.ert().getObservations().getObservationsVector(key).getDataKey()
            if key == data_key:
                keys.append(key)
            else:
                keys.append("%s [%s]" % (key, data_key))



        obs_keys = [observation for observation in gen_obs] + keys
        return sorted(obs_keys, key=lambda k : k.lower())






