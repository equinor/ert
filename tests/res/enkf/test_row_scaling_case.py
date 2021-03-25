#  Copyright (C) 2020  Equinor ASA, Norway.
#
#  The file 'test_row_scaling.py' is part of ERT - Ensemble based Reservoir Tool.
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
import os
import math
import numpy as np
import tempfile
import shutil

from tests import ResTest
from tests.utils import tmpdir

from ecl.util.util import BoolVector
from ecl.grid import EclGridGenerator

from res.test import ErtTestContext
from res.enkf import RowScaling, ResConfig
from res.enkf import EnkfNode, NodeId
from res.enkf import ESUpdate
from res.enkf import FieldConfig, ErtRunContext
from res.enkf.enums import RealizationStateEnum
from res.enkf import EnKFMain

# This function will initialize the data in the case before the actual row
# scaling test can be performed. The function will initialize the PORO field
# which is the parameter to be updated, and it will assign one value for the
# summary value WBHP which will served as simulated response in the update
# process.
#
# The parameter field PORO is initialized by first creating input files
# poro/poro%d.grdecl and then the normal main.initRun() function is used to
# initialize the case and load the PORO data from disk. The summary data is
# normally loaded from a summary case found on disk, here we instead explicitly
# assign a value to the WBHP field at report step 1. Since we circumvent the
# normal load from forward model functionality we must manually update the
# state_map of the file system with the RealizationStateEnum.STATE_HAS_DATA
# flag. The return value from this function is the enkf_fs instance which has
# been initialized.


def init_data(main):
    fsm = main.getEnkfFsManager()
    init_fs = fsm.getFileSystem("init")
    grid = main.eclConfig().getGrid()

    # Model: bhp = poro * 1000
    poro_mean = 0.15
    poro_std = 0.10
    bhp_std = 125

    #Model: wct = poro * 4
    wct_std = 0.30

    bhp = []
    wct = []
    num_realisations = main.getEnsembleSize()

    # The path fields/poro{}.grdecl must be consistent with the INIT_FILES: argument in the
    # PORO configuration in the configuration file used for the testcase.
    os.mkdir("fields")
    random.seed(12345)
    for i in range(num_realisations):
        with open("fields/poro{}.grdecl".format(i), "w") as f:
            poro = random.gauss(poro_mean, poro_std)
            f.write("PORO")
            for i in range(grid.get_num_active()):
                if i % 10 == 0:
                    f.write("\n")

                f.write("{:<7.5} ".format(poro))
            f.write("\n/\n")
        bhp.append(poro * 1000 + random.gauss(0, bhp_std))
        wct.append(poro * 4 + random.gauss(0, wct_std))

    mask = BoolVector(initial_size=main.getEnsembleSize(), default_value=True)
    init_context = ErtRunContext.case_init(init_fs, mask)
    main.initRun(init_context)

    ens_config = main.ensembleConfig()
    bhp_config = ens_config["WBHP"]
    wct_config = ens_config["WWCT"]
    state_map = init_fs.getStateMap()
    for iens in range(main.getEnsembleSize()):
        bhp_node = EnkfNode(bhp_config)
        bhp_summary = bhp_node.as_summary()
        bhp_summary[1] = bhp[iens]

        wct_node = EnkfNode(wct_config)
        wct_summary = wct_node.as_summary()
        wct_summary[1] = wct[iens]

        node_id = NodeId(1, iens)
        bhp_node.save(init_fs, node_id)
        wct_node.save(init_fs, node_id)
        state_map[iens] = RealizationStateEnum.STATE_HAS_DATA

    return init_fs


# This is an example callable which decays as a gaussian away from a position
# obs_pos; this is (probaly) a simplified version of tapering function which
# will be interesting to use.
class GaussianDecay(object):
    def __init__(self, obs_pos, length_scale, grid):
        self.obs_pos = obs_pos
        self.length_scale = length_scale
        self.grid = grid

    def __call__(self, data_index):
        x, y, z = self.grid.get_xyz(active_index=data_index)
        dx = (self.obs_pos[0] - x) / self.length_scale[0]
        dy = (self.obs_pos[1] - y) / self.length_scale[1]
        dz = (self.obs_pos[2] - z) / self.length_scale[2]

        exp_arg = -0.5 * (dx * dx + dy * dy + dz * dz)
        return math.exp(exp_arg)


class SelectLayer(object):
    def __init__(self, layer, grid):
        self.layer = layer
        self.grid = grid

    def __call__(self, data_index):
        ijk = self.grid.get_ijk(active_index=data_index)
        if ijk[2] == self.layer:
            return 1
        else:
            return 0


# This is a quite contrived row scaling callable which is mainly designed to
# test that the correct parts of the field are updated. The behavior of the
# ScalingTest function is as follows:
#
# k = 0: In the upper layer the scaling fcuntion behaves as a step function,
#        for i <= 5 the scaling function returns one, and the updated field
#        should be identical to the field updated without row scaling applied.
#
#        For i > 5 the scaling function will return zero - corresponding to
#        zero update. In this case the field returned from the updated process
#        should be identical to the input field.
#
# k = 1: In the lower layer he scaling function is constantly equal to 0.50,
#        that implies the updated values should be between the initial values
#        and the normal full update value.
#
# The function assert_field_update() verifies that the updated field satisfies
# these constraints.


class ScalingTest(object):
    def __init__(self, grid):
        self.grid = grid

    def __call__(self, data_index):
        i, j, k = self.grid.get_ijk(active_index=data_index)
        if k == 0:
            if i <= 5:
                return 1
            else:
                return 0
        else:
            return 0.5


def assert_field_update(grid, init_field, update_field1, update_field2):
    k = 0
    for j in range(grid.ny):
        for i in range(6):
            assert update_field1.ijk_get_double(
                i, j, k
            ) == update_field2.ijk_get_double(i, j, k)

        for i in range(6, grid.nx):
            assert init_field.ijk_get_double(i, j, k) == update_field2.ijk_get_double(
                i, j, k
            )

    k = 1
    for j in range(grid.ny):
        for i in range(grid.nx):
            init = init_field.ijk_get_double(i, j, k)
            update1 = update_field1.ijk_get_double(i, j, k)
            update2 = update_field1.ijk_get_double(i, j, k)
            assert (update2 - init) * (update1 - init) > 0


class RowScalingTest(ResTest):
    def setUp(self):
        self.config_file = self.createTestPath("local/row_scaling/config")

    # The test_update_workflow() applies the row scaling through a workflow,
    # that is probably the way it will be done by users. The test does not
    # really verify the results in any way, but serves to demonstrate how
    # things are connected. The job in workflows/row_scaling_job1.py uses
    # functools.partial() as an alternative to Class::__call__() as callable
    # when the row scaling is applied.
    def test_update_workflow(self):
        with ErtTestContext("row_scaling", self.config_file) as tc:
            main = tc.getErt()
            workflow_list = main.getWorkflowList()
            workflow = workflow_list["ROW_SCALING_WORKFLOW1"]
            self.assertTrue(workflow.run(main))

            init_fs = init_data(main)
            target_fs = main.getEnkfFsManager().getFileSystem("target")

            es_update = ESUpdate(main)
            run_context = ErtRunContext.ensemble_smoother_update(init_fs, target_fs)
            es_update.smootherUpdate(run_context)

    # The test_update_code() applies the row scaling through code inlined in
    # the test, and also uses the GaussianDecay callable class instead of
    # functools.partial() to create a callable for the scaling operation.
    def test_update_code1(self):
        with ErtTestContext("row_scaling", self.config_file) as tc:
            main = tc.getErt()

            local_config = main.getLocalConfig()
            local_config.clear()
            local_data = local_config.createDataset("LOCAL")
            local_data.addNode("PORO")
            obs = local_config.createObsdata("OBSSET_LOCAL")
            obs.addNode("WBHP0")
            obs.addNode("WWCT0")
            ministep = local_config.createMinistep("MINISTEP_LOCAL")
            ministep.attachDataset(local_data)
            ministep.attachObsset(obs)
            updatestep = local_config.getUpdatestep()
            updatestep.attachMinistep(ministep)

            row_scaling = local_data.row_scaling("PORO")
            ens_config = main.ensembleConfig()
            poro_config = ens_config["PORO"]
            field_config = poro_config.getFieldModelConfig()

            # -------------------------------------------------------------------------------------
            grid = main.eclConfig().getGrid()
            obs_pos = grid.get_xyz(ijk=(5, 5, 1))
            length_scale = (2, 1, 0.50)
            row_scaling.assign(
                field_config.get_data_size(), GaussianDecay(obs_pos, length_scale, grid)
            )
            # -------------------------------------------------------------------------------------

            init_fs = init_data(main)
            target_fs = main.getEnkfFsManager().getFileSystem("target")
            es_update = ESUpdate(main)
            run_context = ErtRunContext.ensemble_smoother_update(init_fs, target_fs)
            es_update.smootherUpdate(run_context)

    # This test does two smoother updates, first without row scaling in update1
    # and then afterwards with row scaling in update2. The row scaling function
    # is designed so that it is possible to test the updates results.
    def test_update_code2(self):
        random_seed = "ABCDEFGHIJK0123456"
        with ErtTestContext("row_scaling", self.config_file) as tc:
            main = tc.getErt()
            init_fs = init_data(main)
            update_fs1 = main.getEnkfFsManager().getFileSystem("target1")

            # The first smoother update without row scaling
            es_update = ESUpdate(main)
            run_context = ErtRunContext.ensemble_smoother_update(init_fs, update_fs1)
            rng = main.rng()
            rng.setState(random_seed)
            es_update.smootherUpdate(run_context)

            # Configure the local updates
            local_config = main.getLocalConfig()
            local_config.clear()
            local_data = local_config.createDataset("LOCAL")
            local_data.addNode("PORO")
            obs = local_config.createObsdata("OBSSET_LOCAL")
            obs.addNode("WWCT0")
            obs.addNode("WBHP0")
            ministep = local_config.createMinistep("MINISTEP_LOCAL")
            ministep.attachDataset(local_data)
            ministep.attachObsset(obs)
            updatestep = local_config.getUpdatestep()
            updatestep.attachMinistep(ministep)

            # Apply the row scaling
            row_scaling = local_data.row_scaling("PORO")
            ens_config = main.ensembleConfig()
            poro_config = ens_config["PORO"]
            field_config = poro_config.getFieldModelConfig()
            grid = main.eclConfig().getGrid()
            row_scaling.assign(field_config.get_data_size(), ScalingTest(grid))

            # Second update with row scaling
            update_fs2 = main.getEnkfFsManager().getFileSystem("target2")
            es_update = ESUpdate(main)
            run_context = ErtRunContext.ensemble_smoother_update(init_fs, update_fs2)
            rng.setState(random_seed)
            es_update.smootherUpdate(run_context)

            # Fetch the three values initial, update without row scaling and
            # update with row scaling and verify that the row scaling has been
            # correctly applied.
            init_node = EnkfNode(poro_config)
            update_node1 = EnkfNode(poro_config)
            update_node2 = EnkfNode(poro_config)
            for iens in range(main.getEnsembleSize()):
                node_id = NodeId(0, iens)

                init_node.load(init_fs, node_id)
                update_node1.load(update_fs1, node_id)
                update_node2.load(update_fs2, node_id)

                assert_field_update(
                    grid,
                    init_node.asField(),
                    update_node1.asField(),
                    update_node2.asField(),
                )

    # This test is identical to test_update_code2(), but the row scaling is
    # applied with the function row_scaling.assign_vector() instead of
    # using a callable.
    def test_row_scaling_using_assign_vector(self):
        random_seed = "ABCDEFGHIJK0123456"
        with ErtTestContext("row_scaling", self.config_file) as tc:
            main = tc.getErt()
            init_fs = init_data(main)
            update_fs1 = main.getEnkfFsManager().getFileSystem("target1")

            # The first smoother update without row scaling
            es_update = ESUpdate(main)
            run_context = ErtRunContext.ensemble_smoother_update(init_fs, update_fs1)
            rng = main.rng()
            rng.setState(random_seed)
            es_update.smootherUpdate(run_context)

            # Configure the local updates
            local_config = main.getLocalConfig()
            local_config.clear()
            local_data = local_config.createDataset("LOCAL")
            local_data.addNode("PORO")
            obs = local_config.createObsdata("OBSSET_LOCAL")
            obs.addNode("WWCT0")
            obs.addNode("WBHP0")
            ministep = local_config.createMinistep("MINISTEP_LOCAL")
            ministep.attachDataset(local_data)
            ministep.attachObsset(obs)
            updatestep = local_config.getUpdatestep()
            updatestep.attachMinistep(ministep)

            # Apply the row scaling
            row_scaling = local_data.row_scaling("PORO")
            ens_config = main.ensembleConfig()
            poro_config = ens_config["PORO"]
            field_config = poro_config.getFieldModelConfig()
            grid = main.eclConfig().getGrid()

            scaling = ScalingTest(grid)
            scaling_vector = np.ndarray(
                [field_config.get_data_size()], dtype=np.float32
            )
            for i in range(field_config.get_data_size()):
                scaling_vector[i] = scaling(i)
            row_scaling.assign_vector(scaling_vector)

            # Second update with row scaling
            update_fs2 = main.getEnkfFsManager().getFileSystem("target2")
            es_update = ESUpdate(main)
            run_context = ErtRunContext.ensemble_smoother_update(init_fs, update_fs2)
            rng.setState(random_seed)
            es_update.smootherUpdate(run_context)

            # Fetch the three values initial, update without row scaling and
            # update with row scaling and verify that the row scaling has been
            # correctly applied.
            init_node = EnkfNode(poro_config)
            update_node1 = EnkfNode(poro_config)
            update_node2 = EnkfNode(poro_config)
            for iens in range(main.getEnsembleSize()):
                node_id = NodeId(0, iens)

                init_node.load(init_fs, node_id)
                update_node1.load(update_fs1, node_id)
                update_node2.load(update_fs2, node_id)

                assert_field_update(
                    grid,
                    init_node.asField(),
                    update_node1.asField(),
                    update_node2.asField(),
                )

    # This test has a configuration where the update consists of two ministeps,
    # where the same field is updated in both steps. Because the
    # obs_data_allocE() function uses random state it is difficult to get
    # identical results from one ministep updating everything and two ministeps
    # updating different parts of the field.
    def test_2ministep(self):
        with ErtTestContext("row_scaling", self.config_file) as tc:
            main = tc.getErt()
            init_fs = init_data(main)
            update_fs1 = main.getEnkfFsManager().getFileSystem("target1")

            # The first smoother update without row scaling
            es_update = ESUpdate(main)
            run_context = ErtRunContext.ensemble_smoother_update(init_fs, update_fs1)
            rng = main.rng()
            es_update.smootherUpdate(run_context)

            # Configure the local updates
            local_config = main.getLocalConfig()
            local_config.clear()
            obs = local_config.createObsdata("OBSSET_LOCAL")
            obs.addNode("WBHP0")

            ministep1 = local_config.createMinistep("MINISTEP1")
            local_data1 = local_config.createDataset("LOCAL1")
            local_data1.addNode("PORO")
            row_scaling1 = local_data1.row_scaling("PORO")
            ministep1.attachDataset(local_data1)
            ministep1.attachObsset(obs)

            ministep2 = local_config.createMinistep("MINISTEP2")
            local_data2 = local_config.createDataset("LOCAL2")
            local_data2.addNode("PORO")
            row_scaling2 = local_data2.row_scaling("PORO")
            ministep2.attachDataset(local_data2)
            ministep2.attachObsset(obs)

            updatestep = local_config.getUpdatestep()
            updatestep.attachMinistep(ministep1)
            updatestep.attachMinistep(ministep2)

            # Apply the row scaling
            ens_config = main.ensembleConfig()
            poro_config = ens_config["PORO"]
            field_config = poro_config.getFieldModelConfig()
            grid = main.eclConfig().getGrid()

            row_scaling1.assign(field_config.get_data_size(), SelectLayer(0, grid))
            row_scaling2.assign(field_config.get_data_size(), SelectLayer(1, grid))

            update_fs2 = main.getEnkfFsManager().getFileSystem("target2")
            es_update = ESUpdate(main)
            run_context = ErtRunContext.ensemble_smoother_update(init_fs, update_fs2)
            es_update.smootherUpdate(run_context)

            init_node = EnkfNode(poro_config)
            node1 = EnkfNode(poro_config)
            node2 = EnkfNode(poro_config)
            for iens in range(main.getEnsembleSize()):
                node_id = NodeId(0, iens)

                init_node.load(init_fs, node_id)
                node1.load(update_fs1, node_id)
                node2.load(update_fs2, node_id)

                init_field = init_node.asField()
                field1 = node1.asField()
                field2 = node2.asField()
                for iv, v1, v2 in zip(init_field, field1, field2):
                    assert iv != v1

    # In the enkf_main_update() routine the A matrix is allocated with a
    # default size; and should grow when exposed to a larger node. At a stage
    # there was a bug in code for combination of row_scaling and
    # matrix_resize(). The purpose of this test is to ensure that we create a
    # sufficiently large node to invoke the rescaling.
    @tmpdir()
    def test_large_case(self):
        with open("config", "w") as fp:
            fp.write(
                """NUM_REALIZATIONS 10
GRID             CASE.EGRID
FIELD            PORO    PARAMETER    poro.grdecl INIT_FILES:fields/poro%d.grdecl
SUMMARY          WBHP
OBS_CONFIG       observations.txt
TIME_MAP timemap.txt
"""
            )

        for f in ["timemap.txt", "observations.txt"]:
            src_file = self.createTestPath(os.path.join("local/row_scaling", f))
            shutil.copy(src_file, "./")
        # The grid size must be greater than 250000 (the default matrix size in
        # enkf_main_update())
        grid = EclGridGenerator.create_rectangular((70, 70, 70), (1, 1, 1))
        grid.save_EGRID("CASE.EGRID")
        res_config = ResConfig(user_config_file="config")
        main = EnKFMain(res_config)
        init_fs = init_data(main)

        # Configure the local updates
        local_config = main.getLocalConfig()
        local_config.clear()
        local_data = local_config.createDataset("LOCAL")
        local_data.addNode("PORO")
        obs = local_config.createObsdata("OBSSET_LOCAL")
        obs.addNode("WBHP0")
        ministep = local_config.createMinistep("MINISTEP_LOCAL")
        ministep.attachDataset(local_data)
        ministep.attachObsset(obs)
        updatestep = local_config.getUpdatestep()
        updatestep.attachMinistep(ministep)

        # Apply the row scaling
        row_scaling = local_data.row_scaling("PORO")
        ens_config = main.ensembleConfig()
        poro_config = ens_config["PORO"]
        field_config = poro_config.getFieldModelConfig()
        grid = main.eclConfig().getGrid()
        row_scaling.assign(field_config.get_data_size(), ScalingTest(grid))
        es_update = ESUpdate(main)
        update_fs = main.getEnkfFsManager().getFileSystem("target2")
        run_context = ErtRunContext.ensemble_smoother_update(init_fs, update_fs)
        es_update.smootherUpdate(run_context)
