import pytest

from tests import ResTest
from res.test import ErtTestContext

from res.enkf.data.enkf_node import EnkfNode
from res.enkf.node_id import NodeId


@pytest.mark.equinor_test
class GenDataTest(ResTest):
    def setUp(self):
        self.config_file = self.createTestPath("Equinor/config/with_GEN_DATA/config")

    def test_create(self):
        with ErtTestContext("gen_data_test", self.config_file) as test_context:
            ert = test_context.getErt()
            fs1 = ert.getEnkfFsManager().getCurrentFileSystem()
            config_node = ert.ensembleConfig().getNode("TIMESHIFT")

            data_node = EnkfNode(config_node)
            data_node.tryLoad(fs1, NodeId(60, 0))

            gen_data = data_node.asGenData()
            data = gen_data.getData()

            self.assertEqual(len(data), 2560)
