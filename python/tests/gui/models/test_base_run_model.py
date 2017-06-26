from ecl.test import ExtendedTestCase
from res.enkf import EnKFMain
from res.test import ErtTestContext
from ert_gui.simulation.models import BaseRunModel
from ert_gui import configureErtNotifier

class BaseRunModelTest(ExtendedTestCase):

    def test_instantiation(self):
        config_file = self.createTestPath('local/simple_config/minimum_config')
        with ErtTestContext('kjell', config_file) as work_area:
            ert = work_area.getErt()
            configureErtNotifier(ert, config_file)
            brm = BaseRunModel('kjell' ,ert.get_queue_config( ))
            self.assertFalse(brm.isQueueRunning())
            self.assertTrue(brm.getProgress() >= 0)
