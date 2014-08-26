'''
Created on Aug 22, 2014

@author: perroe
'''

from ert_gui.models import ErtConnector
from ert.enkf.plot import EnsembleGenKWFetcher

class SensitivityStudyParametersModel(ErtConnector):

    def __init__(self):
        super(SensitivityStudyParametersModel, self).__init__()
        self.__gen_kw_params = None


    def setIsIncluded(self, parameter_name, is_included):
        self.__is_included[parameter_name] = is_included

    def setConstantValue(self, parameter_name, constant_value):
        self.__constant_val[parameter_name] = constant_value

    def getParameters(self):
        if self.__gen_kw_params is None:
            keys = EnsembleGenKWFetcher(self.ert()).getSupportedKeys()
            self.__gen_kw_params = sorted(keys, key=lambda k : k.lower())
            
            self.__is_included  = dict((p, True) for p in self.__gen_kw_params)
            self.__constant_val = dict((p, 0.5) for p in self.__gen_kw_params)

        return self.__gen_kw_params

    def getIsIncluded(self, parameter_name):
        return self.__is_included[parameter_name]

    def getConstantValue(self, parameter_name):
        return self.__constant_val[parameter_name]
