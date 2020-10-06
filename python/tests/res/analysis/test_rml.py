#  Copyright (C) 2013  Equinor ASA, Norway.
#
#  The file 'test_analysis_module.py' is part of ERT - Ensemble based Reservoir Tool.
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
import sys

from tests import ResTest
from ecl.util.enums import RngAlgTypeEnum, RngInitModeEnum
from ecl.util.util import BoolVector, RandomNumberGenerator
from res.util import Matrix
from res.analysis import (
    AnalysisModule,
    AnalysisModuleLoadStatusEnum,
    AnalysisModuleOptionsEnum,
)
from res.enkf import MeasData, ObsData

from tests.utils import tmpdir


def forward_model(params, model_error=False):
    # Y = A*p[0] + B*p[1]
    A = 2
    B = -1
    C = 1
    D = 0.1
    state = [A * params[0] + B * params[1], C * params[0] - D * params[1] * params[0]]
    return state


def measure(state):
    return 0.25 * state[0] - 0.1 * state[1] * state[1]


def init_matrices(ens, mask, obs, rng):
    state_size = 2
    report_step = 5
    meas_data = MeasData(mask)
    meas_block = meas_data.addBlock("OBS", report_step, len(obs))

    A = Matrix(state_size, mask.countEqual(True))
    active_iens = 0
    for iens, params in enumerate(ens):
        if mask[iens]:
            state = forward_model(params)
            meas_block[0, iens] = measure(state)

            A[0, active_iens] = params[0]
            A[1, active_iens] = params[1]

            active_iens += 1

    S = meas_data.createS()

    obs_data = ObsData()
    obs_block = obs_data.addBlock("OBS", 1)
    for iobs, obs_value in enumerate(obs):
        obs_block[iobs] = obs_value

    R = obs_data.createR()
    dObs = obs_data.createDObs()
    E = obs_data.createE(rng, meas_data.getActiveEnsSize())
    D = obs_data.createD(E, S)

    obs_data.scale(S, E=E, D=D, R=R, D_obs=dObs)
    return (A, S, E, D, R, dObs)


class RMLTest(ResTest):
    def setUp(self):
        if sys.platform.lower() == "darwin":
            self.libname = "rml_enkf.dylib"
        else:
            self.libname = "rml_enkf.so"
        self.user = "TEST"

    def createAnalysisModule(self):
        rng = RandomNumberGenerator(RngAlgTypeEnum.MZRAN, RngInitModeEnum.INIT_DEFAULT)
        return AnalysisModule(lib_name=self.libname)

    def test_load_status_enum(self):
        source_file_path = "lib/include/ert/analysis/analysis_module.hpp"
        self.assertEnumIsFullyDefined(
            AnalysisModuleLoadStatusEnum,
            "analysis_module_load_status_enum",
            source_file_path,
        )

    @tmpdir()
    def test_analysis_module(self):

        rng = RandomNumberGenerator()
        module = self.createAnalysisModule()
        ens_size = 12
        obs_size = 1
        state_size = 2

        true_params = [1.25, 0.75]
        true_state = forward_model(true_params)
        obs = [(measure(true_state), 0.75)]
        A = Matrix(state_size, ens_size)

        ens = []
        for iens in range(ens_size):
            param = [random.gauss(1.00, 1.00), random.gauss(1.00, 1.00)]
            ens.append(param)

        ens_mask = BoolVector(default_value=True, initial_size=ens_size)
        ens_mask[2] = False
        obs_mask = BoolVector()
        obs_mask[0] = 1
        (A, S, E, D, R, dObs) = init_matrices(ens, ens_mask, obs, rng)

        module.initUpdate(ens_mask, obs_mask, S, R, dObs, E, D, rng)
        module.updateA(A, S, R, dObs, E, D, rng)

        ens_mask[10] = False
        ens_mask[5] = False
        (A, S, E, D, R, dObs) = init_matrices(ens, ens_mask, obs, rng)
        self.assertEqual(S.dims(), (obs_size, ens_mask.countEqual(True)))
        self.assertEqual(E.dims(), (obs_size, ens_mask.countEqual(True)))
        self.assertEqual(D.dims(), (obs_size, ens_mask.countEqual(True)))

        module.initUpdate(ens_mask, obs_mask, S, R, dObs, E, D, rng)
        module.updateA(A, S, R, dObs, E, D, rng)
