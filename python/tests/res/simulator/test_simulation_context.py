import time
from tests import ResTest
from tests.utils import tmpdir
from ecl.util.util import BoolVector

from res.test import ErtTestContext
from tests.utils import wait_until
from res.enkf.enums import RealizationStateEnum
from res.simulator import SimulationContext


class SimulationContextTest(ResTest):

    @tmpdir()
    def test_simulation_context(self):
        config_file = self.createTestPath("local/snake_oil_no_data/snake_oil.ert")
        with ErtTestContext("ert/server/rpc/simulation_context", config_file) as test_context:
            ert = test_context.getErt()

            size = 4
            mask1 = BoolVector( initial_size = size )
            mask2 = BoolVector( initial_size = size )

            for iens_2 in range(size//2):
                mask1[2*iens_2] = True
                mask1[2*iens_2 + 1] = False

                mask2[2*iens_2] = False
                mask2[2*iens_2 + 1] = True

            fs_manager = ert.getEnkfFsManager()
            first_half = fs_manager.getFileSystem("first_half")
            other_half = fs_manager.getFileSystem("other_half")

            # i represents geo_id
            case_data = [(i,{}) for i in range(size)]
            simulation_context1 = SimulationContext(ert, first_half, mask1 , 0, case_data)
            simulation_context2 = SimulationContext(ert, other_half, mask2 , 0, case_data)

            for iens in range(size):
                if iens % 2 == 0:
                    self.assertFalse(simulation_context1.isRealizationFinished(iens))
                    # do we have the proper geo_id in run_args?
                    self.assertEqual(simulation_context1.get_run_args(iens).geo_id, iens)
                else:
                    self.assertFalse(simulation_context2.isRealizationFinished(iens))
                    self.assertEqual(simulation_context2.get_run_args(iens).geo_id, iens)

            wait_until(
                func=(lambda: self.assertFalse(simulation_context1.isRunning() or simulation_context2.isRunning())),
                timeout=90
            )

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
