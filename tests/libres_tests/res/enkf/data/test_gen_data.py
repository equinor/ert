from libres_utils import ResTest

from res.enkf.data.enkf_node import EnkfNode
from res.enkf.node_id import NodeId
from res.test import ErtTestContext


class GenDataTest(ResTest):
    def setUp(self):
        self.config_file = self.createTestPath("local/snake_oil/snake_oil.ert")

    def test_create(self):
        with ErtTestContext(self.config_file) as test_context:
            ert = test_context.getErt()
            fs1 = ert.getEnkfFsManager().getCurrentFileSystem()
            config_node = ert.ensembleConfig().getNode("SNAKE_OIL_OPR_DIFF")

            data_node = EnkfNode(config_node)
            data_node.tryLoad(fs1, NodeId(199, 0))

            gen_data = data_node.asGenData()
            data = gen_data.getData()

            self.assertEqual(2000, len(data))
