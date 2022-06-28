from libres_utils import ResTest, tmpdir, wait_until

from res.enkf.enums import RealizationStateEnum
from res.simulator import SimulationContext
from res.test import ErtTestContext


class SimulationContextTest(ResTest):
    @tmpdir()
    def test_simulation_context(self):
        config_file = self.createTestPath("local/batch_sim/sleepy_time.ert")
        with ErtTestContext(config_file) as test_context:
            ert = test_context.getErt()

            size = 4
            even_mask = [True, False] * (size // 2)
            odd_mask = [False, True] * (size // 2)

            fs_manager = ert.getEnkfFsManager()
            even_half = fs_manager.getFileSystem("even_half")
            odd_half = fs_manager.getFileSystem("odd_half")

            case_data = [(geo_id, {}) for geo_id in range(size)]
            even_ctx = SimulationContext(ert, even_half, even_mask, 0, case_data)
            odd_ctx = SimulationContext(ert, odd_half, odd_mask, 0, case_data)

            for iens in range(size):
                if iens % 2 == 0:
                    self.assertFalse(even_ctx.isRealizationFinished(iens))
                else:
                    self.assertFalse(odd_ctx.isRealizationFinished(iens))

            def any_is_running():
                return even_ctx.isRunning() or odd_ctx.isRunning()

            wait_until(func=(lambda: self.assertFalse(any_is_running())), timeout=90)

            self.assertEqual(even_ctx.getNumFailed(), 0)
            self.assertEqual(even_ctx.getNumRunning(), 0)
            self.assertEqual(even_ctx.getNumSuccess(), size / 2)

            self.assertEqual(odd_ctx.getNumFailed(), 0)
            self.assertEqual(odd_ctx.getNumRunning(), 0)
            self.assertEqual(odd_ctx.getNumSuccess(), size / 2)

            even_state_map = even_half.getStateMap()
            odd_state_map = odd_half.getStateMap()

            for iens in range(size):
                if iens % 2 == 0:
                    self.assertTrue(even_ctx.didRealizationSucceed(iens))
                    self.assertFalse(even_ctx.didRealizationFail(iens))
                    self.assertTrue(even_ctx.isRealizationFinished(iens))

                    self.assertEqual(
                        even_state_map[iens], RealizationStateEnum.STATE_HAS_DATA
                    )
                else:
                    self.assertTrue(odd_ctx.didRealizationSucceed(iens))
                    self.assertFalse(odd_ctx.didRealizationFail(iens))
                    self.assertTrue(odd_ctx.isRealizationFinished(iens))

                    self.assertEqual(
                        odd_state_map[iens], RealizationStateEnum.STATE_HAS_DATA
                    )
