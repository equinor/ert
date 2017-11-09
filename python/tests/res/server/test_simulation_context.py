import time
import os.path
import sys
from ecl.test import ExtendedTestCase
from ecl.util import BoolVector

from res.test import ErtTestContext
from res.enkf import EnkfVarType
from res.enkf.enums import RealizationStateEnum
from res.server import SimulationContext


class SimulationContextTest(ExtendedTestCase):


    def test_simulation_context(self):
        config_file = self.createTestPath("local/snake_oil_no_data/snake_oil.ert")
        with ErtTestContext("ert/server/rpc/simulation_context", config_file) as test_context:
            ert = test_context.getErt()

            size = 4
            mask1 = BoolVector( initial_size = size )
            mask2 = BoolVector( initial_size = size )

            for iens_2 in range(size/2):
                mask1[2*iens_2] = True
                mask1[2*iens_2 + 1] = False

                mask2[2*iens_2] = False
                mask2[2*iens_2 + 1] = True


            fs_manager = ert.getEnkfFsManager()
            first_half = fs_manager.getFileSystem("first_half")
            other_half = fs_manager.getFileSystem("other_half")

            simulation_context1 = SimulationContext(ert, first_half, mask1 , 0)
            simulation_context2 = SimulationContext(ert, other_half, mask2 , 0)

            ert.createRunpath( simulation_context1.get_run_context( ) )
            ert.createRunpath( simulation_context2.get_run_context( ) )

            geo_id = 0
            for iens in range(size):
                if iens % 2 == 0:
                    simulation_context1.addSimulation(iens, geo_id)
                    self.assertFalse(simulation_context1.isRealizationFinished(iens))
                else:
                    simulation_context2.addSimulation(iens, geo_id)
                    self.assertFalse(simulation_context2.isRealizationFinished(iens))


            with self.assertRaises(UserWarning):
                simulation_context1.addSimulation(size, geo_id)

            with self.assertRaises(UserWarning):
                simulation_context1.addSimulation(0, geo_id)

            while simulation_context1.isRunning():
                time.sleep(1.0)

            while simulation_context2.isRunning():
                time.sleep(1.0)

            self.assertEqual(simulation_context1.getNumFailed(), 0)
            self.assertEqual(simulation_context1.getNumRunning(), 0)
            self.assertEqual(simulation_context1.getNumSuccess(), size/2)

            self.assertEqual(simulation_context2.getNumFailed(), 0)
            self.assertEqual(simulation_context2.getNumRunning(), 0)
            self.assertEqual(simulation_context2.getNumSuccess(), size/2)

            first_half_state_map = first_half.getStateMap()
            other_half_state_map = other_half.getStateMap()

            for iens in range(size):
                if iens % 2 == 0:
                    self.assertTrue(simulation_context1.didRealizationSucceed(iens))
                    self.assertFalse(simulation_context1.didRealizationFail(iens))
                    self.assertTrue(simulation_context1.isRealizationFinished(iens))

                    self.assertEqual(first_half_state_map[iens], RealizationStateEnum.STATE_HAS_DATA)
                else:
                    self.assertTrue(simulation_context2.didRealizationSucceed(iens))
                    self.assertFalse(simulation_context2.didRealizationFail(iens))
                    self.assertTrue(simulation_context2.isRealizationFinished(iens))

                    self.assertEqual(other_half_state_map[iens], RealizationStateEnum.STATE_HAS_DATA)
