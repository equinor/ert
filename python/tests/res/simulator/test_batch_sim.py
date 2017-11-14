import os
import stat
import time
import sys
import unittest
from ecl.test import ExtendedTestCase, TestAreaContext
from ecl.util import BoolVector

from res.test import ErtTestContext
from res.server import SimulationContext
from res.simulator import BatchSimulator
from res.enkf import ResConfig
from tests.res.enkf.test_programmatic_res_config import ProgrammaticResConfigTest as ResConfigTest



class BatchSimulatorTest(ExtendedTestCase):


    def test_create_simulator(self):
        config_file = self.createTestPath("local/batch_sim/batch_sim.ert")
        with TestAreaContext("batch_sim", store_area = True) as ta:
            sys.stderr.write("cwd:%s " % os.getcwd())
            ta.copy_parent_content( config_file )

            # Not valid ResConfig instance as first argument
            with self.assertRaises(ValueError):
                rsim = BatchSimulator( "ARG",
                                       {"WELL_ORDER" : ["W1", "W2", "W3"],
                                        "WELL_ON_OFF" : ["W1","W2", "W3"]},
                                       ["ORDER", "ON_OFF"])

            res_config = ResConfig( user_config_file = os.path.basename( config_file ))
            # Control argument not a dict - Exception
            with self.assertRaises(Exception):
                rsim = BatchSimulator(res_config, ["WELL_ORDER", ["W1","W2","W3"]], ["ORDER"])

            rsim = BatchSimulator( res_config,
                                   {"WELL_ORDER" : ["W1", "W2", "W3"],
                                    "WELL_ON_OFF" : ["W1","W2", "W3"]},
                                   ["ORDER", "ON_OFF"])

            # The key for one of the controls is invalid => KeyError
            with self.assertRaises(KeyError):
                rsim.start("case", [ (2, {"WELL_ORDERX" : [0,0,1], "WELL_ON_OFF" : [0,0,1]}),
                                     (1, {"WELL_ORDER" : [0,0,0] , "WELL_ON_OFF" : [0,0,1]}) ])


            # Missing the key WELL_ON_OFF => ValueError
            with self.assertRaises(ValueError):
                rsim.start("case", [ (2, {"WELL_ORDER" : [0,0,1]}) ])

            # One of the numeric vectors has wrong length => ValueError:
            with self.assertRaises(ValueError):
                rsim.start("case", [ (2, {"WELL_ORDER" : [0,0,1], "WELL_ON_OFF" : [0]}) ])

            # Not numeric values => Exception
            with self.assertRaises(Exception):
                rsim.start("case", [ (2, {"WELL_ORDER" : [0,0,1], "WELL_ON_OFF" : [0,1,'X']}) ])

            # Not numeric values => Exception
            with self.assertRaises(Exception):
               rsim.start("case", [ ('2', {"WELL_ORDER" : [0,0,1], "WELL_ON_OFF" : [0,1,4]}) ])


            # Starting a simulation which should actually run through.
            ctx = rsim.start("case", [(2, {"WELL_ORDER" : [1, 2, 3], "WELL_ON_OFF" : [4,5,6]}),
                                      (1, {"WELL_ORDER" : [7, 8, 9], "WELL_ON_OFF" : [10,11,12]})])

            # Asking for results before it is complete.
            with self.assertRaises(RuntimeError):
                res = ctx.results()


            while ctx.running():
                status = ctx.status
                time.sleep(1)
                sys.stderr.write("status: %s\n" % str(status))

            res = ctx.results()
            self.assertEqual(len(res), 2)
            res0 = res[0]
            res1 = res[1]
            self.assertIn("ORDER", res0)
            self.assertIn("ON_OFF", res1)


            # The forward model job SQUARE_PARAMS will load the control values and square them
            # before writing results to disk.
            order0 = res0["ORDER"]
            for i,x in enumerate(range(1,4)):
                self.assertEqual(order0[i], x*x)

            on_off0 = res0["ON_OFF"]
            for i,x in enumerate(range(4,7)):
                self.assertEqual(on_off0[i], x*x)

            order1 = res1["ORDER"]
            for i,x in enumerate(range(7,10)):
                self.assertEqual(order1[i], x*x)

            on_off1 = res1["ON_OFF"]
            for i,x in enumerate(range(10,13)):
                self.assertEqual(on_off1[i], x*x)


if __name__ == "__main__":
    unittest.main()
