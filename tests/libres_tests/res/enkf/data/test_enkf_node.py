import json

from ecl.util.test import TestAreaContext
from ....libres_utils import ResTest

from res.enkf.config import EnkfConfigNode
from res.enkf.data import EnkfNode


class EnkfNodeTest(ResTest):
    def test_config(self):
        keys = ["Key1", "Key2", "Key3"]
        with TestAreaContext("enkf_node"):
            config = EnkfConfigNode.create_ext_param("key", keys)
            node = EnkfNode(config)
            ext_node = node.as_ext_param()
            config.getModelConfig()

            ext_node.set_vector([1, 2, 3])
            node.ecl_write("path")
            d = json.load(open("path/key.json"))
            self.assertEqual(d["Key1"], 1)
            self.assertEqual(d["Key3"], 3)
