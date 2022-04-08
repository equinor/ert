from libres_utils import ResTest, tmpdir

from res.test import ErtTestContext


class EnKFFSManagerTest1(ResTest):
    def setUp(self):
        self.config_file = self.createTestPath("local/snake_oil/snake_oil.ert")

    @tmpdir()
    def test_create(self):
        # We are indirectly testing the create through the create
        # already in the enkf_main object. In principle we could
        # create a separate manager instance from the ground up, but
        # then the reference count will be weird.
        with ErtTestContext(
            "enkf_fs_manager_create_test", self.config_file
        ) as testContext:
            ert = testContext.getErt()
            fsm = ert.getEnkfFsManager()

            fsm.getCurrentFileSystem()
            self.assertTrue(fsm.isCaseMounted("default_0"))
            self.assertTrue(fsm.caseExists("default_0"))
            self.assertTrue(fsm.caseHasData("default_0"))
            self.assertFalse(fsm.isCaseRunning("default_0"))

            self.assertEqual(1, fsm.getFileSystemCount())

            self.assertFalse(fsm.isCaseMounted("newFS"))
            self.assertFalse(fsm.caseExists("newFS"))
            self.assertFalse(fsm.caseHasData("newFS"))
            self.assertFalse(fsm.isCaseRunning("newFS"))

            fsm.getFileSystem("newFS")
            self.assertEqual(2, fsm.getFileSystemCount())

            self.assertTrue(fsm.isCaseMounted("newFS"))
            self.assertTrue(fsm.caseExists("newFS"))
            self.assertFalse(fsm.caseHasData("newFS"))
            self.assertFalse(fsm.isCaseRunning("newFS"))
