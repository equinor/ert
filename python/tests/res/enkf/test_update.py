#  Copyright (C) 2015  Equinor ASA, Norway.
#
#  The file 'test_update.py' is part of ERT - Ensemble based Reservoir Tool.
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

import random
from tests import ResTest
from res.test import ErtTestContext

from ecl.util.enums import RngAlgTypeEnum, RngInitModeEnum
from ecl.util import Matrix, BoolVector , RandomNumberGenerator
from res.analysis import AnalysisModule, AnalysisModuleLoadStatusEnum, AnalysisModuleOptionsEnum
from res.enkf import MeasData, ObsData, LocalObsdata


def update(rng , mask , module , ert , meas_data , obs_data , state_size):
    S = meas_data.createS()
    R = obs_data.createR()
    dObs = obs_data.createDObs()
    E = obs_data.createE( rng , meas_data.getActiveEnsSize() )
    D = obs_data.createD(E , S)
    obs_data.scale(S , E = E , D = D , R = R , D_obs = dObs)

    A = Matrix(state_size , meas_data.getActiveEnsSize())
    A.randomInit( rng )

    module.initUpdate( mask , S , R , dObs , E , D )
    module.updateA( A , S , R , dObs , E , D )




class UpdateTest(ResTest):
  def setUp(self):
      self.libname = ert.ert_lib_path + "/rml_enkf.so"
      self.config_file = self.createTestPath("Equinor/config/obs_testing2/config")
      self.rng = RandomNumberGenerator(RngAlgTypeEnum.MZRAN, RngInitModeEnum.INIT_DEFAULT)


  def createAnalysisModule(self):
      return AnalysisModule(self.rng, lib_name = self.libname)


  def test_it(self):
      state_size = 10
      with ErtTestContext("update" , self.config_file) as tc:
          analysis = self.createAnalysisModule()
          ert = tc.getErt()
          obs = ert.getObservations()
          local_obsdata = obs.getAllActiveLocalObsdata( )

          fs = ert.getEnkfFsManager().getCurrentFileSystem()


          mask = BoolVector( initial_size = ert.getEnsembleSize() , default_value = True)
          meas_data = MeasData(mask)
          obs_data = ObsData()
          obs.getObservationAndMeasureData( fs , local_obsdata , mask.createActiveList() , meas_data , obs_data )
          update( self.rng , mask , analysis , ert , meas_data , obs_data , state_size)


          mask[0] = False
          mask[4] = False
          meas_data = MeasData(mask)
          obs_data = ObsData()
          obs.getObservationAndMeasureData( fs , local_obsdata , mask.createActiveList() , meas_data , obs_data )
          update( self.rng , mask , analysis , ert , meas_data , obs_data , state_size)


