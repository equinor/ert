import os.path
import json

from ecl.util.test.ecl_mock import createEclSum
from res.enkf.data import Summary
from res.enkf.config import SummaryConfig
from ecl.util.test import TestAreaContext
from tests import ResTest


class SummaryTest(ResTest):
    def test_create(self):
        config = SummaryConfig("WWCT:OP_5")
        summary = Summary(config)
        self.assertEqual(len(summary), 0)

        with self.assertRaises(IndexError):
            v = summary[100]
