import pytest
from ecl.summary import EclSum
from ecl.util.test import TestAreaContext
from ...libres_utils import ResTest, tmpdir

from ert._c_wrappers.enkf import (
    AnalysisConfig,
    EclConfig,
    EnkfConfigNode,
    EnKFMain,
    ErtTemplate,
    ResConfig,
)
from ert._c_wrappers.enkf.util import TimeMap


@pytest.mark.unstable
class EnKFLibraryTest(ResTest):
    def setUp(self):
        self.case_directory = self.createTestPath("local/simple_config/")

    def test_failed_class_creation(self):
        classes = [EnkfConfigNode, ErtTemplate]

        for cls in classes:
            with self.assertRaises(NotImplementedError):
                cls()

    @tmpdir()
    def test_ecl_config_creation(self):
        with TestAreaContext("enkf_library_test") as work_area:
            work_area.copy_directory(self.case_directory)

            res_config = ResConfig("simple_config/minimum_config")
            main = EnKFMain(res_config)

            self.assertIsInstance(main.analysisConfig(), AnalysisConfig)
            self.assertIsInstance(main.eclConfig(), EclConfig)

            with self.assertRaises(AssertionError):  # Null pointer!
                self.assertIsInstance(main.eclConfig().getRefcase(), EclSum)

            file_system = main.getEnkfFsManager().getCurrentFileSystem()
            self.assertEqual(file_system.getCaseName(), "default")
            time_map = file_system.getTimeMap()
            self.assertIsInstance(time_map, TimeMap)
