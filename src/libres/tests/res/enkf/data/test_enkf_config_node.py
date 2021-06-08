import os.path
import json

from res.enkf.config import EnkfConfigNode
from ecl.util.test import TestAreaContext
from tests import ResTest


class EnkfConfigNodeTest(ResTest):
    def test_gen_data(self):

        # Must have %d in filename argument
        with self.assertRaises(ValueError):
            config_node = EnkfConfigNode.create_gen_data("KEY", "FILE")

        config_node = EnkfConfigNode.create_gen_data("KEY", "FILE%d")
        self.assertIsInstance(config_node, EnkfConfigNode)
        gen_data = config_node.getModelConfig()
        self.assertEqual(1, gen_data.getNumReportStep())
        self.assertEqual(0, gen_data.getReportStep(0))

        config_node = EnkfConfigNode.create_gen_data(
            "KEY", "FILE%d", report_steps=[10, 20, 30]
        )
        self.assertIsInstance(config_node, EnkfConfigNode)
        gen_data = config_node.getModelConfig()
        self.assertEqual(3, gen_data.getNumReportStep())
        for r1, r2 in zip([10, 20, 30], gen_data.getReportSteps()):
            self.assertEqual(r1, r2)
