#  Copyright (C) 2017  Equinor ASA, Norway.
#
#  The file 'test_enkf_sim_model.py' is part of ERT - Ensemble based Reservoir Tool.
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

import os
import sys
import json
import subprocess

from ecl.util.test import TestAreaContext
from tests import ResTest
from ecl.util.util import BoolVector

from res.job_queue import ExtJob

from res.enkf import (EnsembleConfig, AnalysisConfig, ModelConfig, SiteConfig,
                      EclConfig, PlotSettings, EnkfObs, ErtTemplates, EnkfFs,
                      EnKFState, EnkfVarType, ObsVector, RunArg, ResConfig)
from res.enkf.config import EnkfConfigNode
from res.enkf.enkf_main import EnKFMain
from res.enkf.ert_run_context import ErtRunContext
from res.enkf.enums import (EnkfObservationImplementationType, LoadFailTypeEnum,
                            EnkfInitModeEnum, ErtImplType, RealizationStateEnum,
                            EnkfRunType, EnkfFieldFileFormatEnum,
                            EnkfTruncationType, ActiveMode)

from res.enkf.observations.summary_observation import SummaryObservation
from res.test import ErtTestContext


class EnKFTestSimModel(ResTest):

  def setUp(self):
    pass

  def test_simulation_model(self):

    with TestAreaContext('enkf_test_sim_model_kw') as work_area:
      base_path = os.getcwd()
      source_path = self.createTestPath('local/simulation_model')

      work_area.copy_directory(source_path)
      dir_ert = os.path.join(base_path, 'simulation_model');
      assert(os.path.isdir(  dir_ert  )  )

      file_ert = os.path.join(dir_ert, 'sim_kw.ert')
      assert(  os.path.isfile(file_ert)  )

      with ErtTestContext( "sim_kw", model_config = file_ert, store_area = True) as ctx:
        ert = ctx.getErt( )
        fs_manager = ert.getEnkfFsManager()
        result_fs = fs_manager.getCurrentFileSystem( )

        model_config = ert.getModelConfig( )
        forward_model = model_config.getForwardModel()
        self.assertEqual(forward_model.get_size(), 4)
        self.assertEqual( forward_model.iget_job(3).get_arglist(), ['WORD_A'] )
        self.assertEqual( forward_model.iget_job(0).get_arglist(), ['<ARGUMENT>'] )
        self.assertEqual( forward_model.iget_job(1).get_arglist(), ['Hello', 'True',        '3.14', '4'] )
        self.assertEqual( forward_model.iget_job(2).get_arglist(), ['word',  '<ECLBASE>'] )

        runpath_fmt = model_config.getRunpathFormat( )
        jobname_fmt = model_config.getJobnameFormat( )

        subst_list = ert.getDataKW( )
        itr = 0
        mask = BoolVector( default_value = True, initial_size = 1 )

        run_context = ErtRunContext.ensemble_experiment( result_fs, mask, runpath_fmt, jobname_fmt, subst_list, itr)
        ert.getEnkfSimulationRunner().createRunPath( run_context )
        queue_config = ert.get_queue_config()
        self.assertEqual( queue_config.num_cpu , 5 )
        os.chdir('storage/sim_kw/runpath/realisation-0/iter-0')
        assert(   os.path.isfile('jobs.json')   )
        with open("jobs.json", "r") as f:
          data = json.load(f)
          jobList = data["jobList"]
          old_job_A = jobList[3]
          self.assertEqual(old_job_A["argList"], ['WORD_A'])
          old_job_B = jobList[0]
          self.assertEqual(old_job_B["argList"], ['yy'])
          new_job_A = jobList[1]
          self.assertEqual(new_job_A["argList"], ['Hello', 'True', '3.14', '4'])
          new_job_B = jobList[2]
          self.assertEqual(new_job_B["argList"], ['word', 'SIM_KW'])
          
          
        












