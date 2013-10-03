#  Copyright (C) 2012 Statoil ASA, Norway.
#
#  The file 'summary_observation.py' is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.

from ert.cwrap import BaseCClass, CWrapper
from ert.enkf import ENKF_LIB


class SummaryObservation(BaseCClass):
    def __init__(self):
        raise NotImplementedError("Class can not be instantiated directly!")

    def getValue(self):
        """ @rtype: float """
        return SummaryObservation.cNamespace().get_value(self)

    def getStandardDeviation(self):
        """ @rtype: float """
        return SummaryObservation.cNamespace().get_std(self)

    def getSummaryKey(self):
        """ @rtype: str """
        return SummaryObservation.cNamespace().get_summary_key(self)



cwrapper = CWrapper(ENKF_LIB)
cwrapper.registerType("summary_obs", SummaryObservation)
cwrapper.registerType("summary_obs_obj", SummaryObservation.createPythonObject)
cwrapper.registerType("summary_obs_ref", SummaryObservation.createCReference)

SummaryObservation.cNamespace().get_value = cwrapper.prototype("double summary_obs_get_value(summary_obs)")
SummaryObservation.cNamespace().get_std = cwrapper.prototype("double summary_obs_get_std(summary_obs)")
SummaryObservation.cNamespace().get_summary_key = cwrapper.prototype("char* summary_obs_get_summary_key(summary_obs)")