from ert.cwrap import clib, CWrapper
from ert.enkf.data.enkf_node import EnkfNode
from ert.enkf.enums.enkf_state_type_enum import EnkfStateType
from ert.enkf.node_id import NodeId
from ert.test import ErtTestContext
from ert.test.extended_testcase import ExtendedTestCase

test_lib  = clib.ert_load("libenkf")
cwrapper =  CWrapper(test_lib)

get_active_mask = cwrapper.prototype("bool_vector_ref gen_data_config_get_active_mask( gen_data_config )")

class GenDataConfigTest(ExtendedTestCase):
    def setUp(self):
        self.config_file = self.createTestPath("Statoil/config/with_GEN_DATA/config")

    def load_active_masks(self, case1, case2 ):
        with ErtTestContext("gen_data_config_test", self.config_file) as test_context:
            ert = test_context.getErt()

            fs =  ert.getEnkfFsManager().getFileSystem(case1)
            config_node = ert.ensembleConfig().getNode("TIMESHIFT")
            data_node = EnkfNode(config_node)
            data_node.tryLoad(fs, NodeId(60, 0, EnkfStateType.FORECAST))

            first_active_mask = get_active_mask( config_node.getDataModelConfig() )
            first_active_mask_length = len(first_active_mask)
            self.assertEqual(first_active_mask_length, 2560)

            fs =  ert.getEnkfFsManager().getFileSystem(case2)
            data_node = EnkfNode(config_node)
            data_node.tryLoad(fs, NodeId(60, 0, EnkfStateType.FORECAST))

            second_active_mask = get_active_mask( config_node.getDataModelConfig() )
            self.assertEqual(len(second_active_mask), 2560)

            self.assertEqual(first_active_mask_length, len(second_active_mask))


    def test_loading_two_cases_with_and_withough_active_file(self):
        self.load_active_masks("default", "missing-active")
        self.load_active_masks("missing-active", "default")







