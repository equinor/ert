from ....libres_utils import ResTest

from ert._c_wrappers.enkf import NodeId
from ert._c_wrappers.enkf.config import GenDataConfig
from ert._c_wrappers.enkf.data import EnkfNode
from ert._c_wrappers.test import ErtTestContext


class GenDataConfigTest(ResTest):
    def setUp(self):
        self.config_file = self.createTestPath("local/snake_oil/snake_oil.ert")

    def load_active_masks(self, case1, case2):
        with ErtTestContext(self.config_file) as test_context:
            ert = test_context.getErt()

            fs1 = ert.getEnkfFsManager().getFileSystem(case1)
            config_node = ert.ensembleConfig().getNode("SNAKE_OIL_OPR_DIFF")
            data_node = EnkfNode(config_node)
            data_node.tryLoad(fs1, NodeId(199, 0))

            active_mask = config_node.getDataModelConfig().getActiveMask()
            first_active_mask_length = len(active_mask)
            self.assertEqual(first_active_mask_length, 2000)

            fs2 = ert.getEnkfFsManager().getFileSystem(case2)
            data_node = EnkfNode(config_node)
            data_node.tryLoad(fs2, NodeId(199, 0))

            active_mask = config_node.getDataModelConfig().getActiveMask()
            second_active_mask_len = len(active_mask)
            self.assertEqual(second_active_mask_len, 2000)
            self.assertEqual(first_active_mask_length, second_active_mask_len)

            # Setting one element to False, load different case, check, reload,
            # and check.
            self.assertTrue(active_mask[10])
            active_mask_modified = active_mask.copy()
            active_mask_modified[10] = False

            # Load first - check element is true
            data_node = EnkfNode(config_node)
            data_node.tryLoad(fs1, NodeId(199, 0))
            active_mask = config_node.getDataModelConfig().getActiveMask()
            self.assertTrue(active_mask[10])

    def test_loading_two_cases_with_and_without_active_file(self):
        self.load_active_masks("default_0", "default_1")

    def test_create(self):
        GenDataConfig("KEY")
