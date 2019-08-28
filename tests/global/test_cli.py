from tests import ErtTest
from res.test import ErtTestContext
import os
import subprocess
from res.enkf import EnKFMain, ResConfig
from ecl.util.util import BoolVector
from ert_gui import cli
from ert_gui import ERT
from ert_gui.cli import ErtCliNotifier
from argparse import Namespace
from ert_gui.simulation.models.ensemble_experiment import EnsembleExperiment
from ert_gui.simulation.models.ensemble_smoother import EnsembleSmoother
from ert_gui.simulation.models.multiple_data_assimilation import \
    MultipleDataAssimilation
from ert_gui.simulation.models.single_test_run import SingleTestRun


class EntryPointTest(ErtTest):

    def test_custom_target_case_name(self):
        config_file = self.createTestPath('local/poly_example/poly.ert')
        with ErtTestContext('test_custom_target_case_name', config_file) as work_area:
            ert = work_area.getErt()
            notifier = ErtCliNotifier(ert, config_file)
            ERT.adapt(notifier)

            custom_name = "test"
            args = Namespace(target_case=custom_name)
            res = cli._target_case_name(args)
            self.assertEqual(custom_name, res)

    def test_default_target_case_name(self):
        config_file = self.createTestPath('local/poly_example/poly.ert')
        with ErtTestContext('test_default_target_case_name', config_file) as work_area:
            ert = work_area.getErt()
            notifier = ErtCliNotifier(ert, config_file)
            ERT.adapt(notifier)

            args = Namespace(target_case=None)
            res = cli._target_case_name(args)
            self.assertEqual("default_smoother_update", res)

    def test_default_target_case_name_format_mode(self):
        config_file = self.createTestPath('local/poly_example/poly.ert')
        with ErtTestContext('test_default_target_case_name_format_mode', config_file) as work_area:
            ert = work_area.getErt()
            notifier = ErtCliNotifier(ert, config_file)
            ERT.adapt(notifier)

            args = Namespace(target_case=None)
            res = cli._target_case_name(args, format_mode=True)
            self.assertEqual("default_%d", res)

    def test_default_realizations(self):
        config_file = self.createTestPath('local/poly_example/poly.ert')
        with ErtTestContext('test_default_realizations', config_file) as work_area:
            ert = work_area.getErt()
            notifier = ErtCliNotifier(ert, config_file)
            ERT.adapt(notifier)

            args = Namespace(realizations=None)
            res = cli._realizations(args)
            ensemble_size = ERT.ert.getEnsembleSize()
            mask = BoolVector(default_value=False, initial_size=ensemble_size)
            mask.updateActiveMask("0-99")
            self.assertEqual(mask, res)

    def test_custom_realizations(self):
        config_file = self.createTestPath('local/poly_example/poly.ert')
        with ErtTestContext('test_custom_realizations', config_file) as work_area:
            ert = work_area.getErt()
            notifier = ErtCliNotifier(ert, config_file)
            ERT.adapt(notifier)

            args = Namespace(realizations="0-4,7,8")
            res = cli._realizations(args)
            ensemble_size = ERT.ert.getEnsembleSize()
            mask = BoolVector(default_value=False, initial_size=ensemble_size)
            mask.updateActiveMask("0-4,7,8")
            self.assertEqual(mask, res)

    def test_setup_single_test_run(self):
        config_file = self.createTestPath('local/poly_example/poly.ert')
        with ErtTestContext('test_single_test_run', config_file) as work_area:
            ert = work_area.getErt()
            notifier = ErtCliNotifier(ert, config_file)
            ERT.adapt(notifier)

            model, argument = cli._setup_single_test_run()
            self.assertTrue(isinstance(model, SingleTestRun))
            self.assertEquals(1, len(argument.keys()))
            self.assertTrue("active_realizations" in argument)

    def test_setup_ensemble_experiment(self):
        config_file = self.createTestPath('local/poly_example/poly.ert')
        with ErtTestContext('test_single_test_run', config_file) as work_area:
            ert = work_area.getErt()
            notifier = ErtCliNotifier(ert, config_file)
            ERT.adapt(notifier)

            model, argument = cli._setup_single_test_run()
            self.assertTrue(isinstance(model, EnsembleExperiment))
            self.assertEquals(1, len(argument.keys()))
            self.assertTrue("active_realizations" in argument)

    def test_setup_ensemble_smoother(self):
        config_file = self.createTestPath('local/poly_example/poly.ert')
        with ErtTestContext('test_single_test_run', config_file) as work_area:
            ert = work_area.getErt()
            notifier = ErtCliNotifier(ert, config_file)
            ERT.adapt(notifier)
            args = Namespace(realizations="0-4,7,8", target_case="test_case")

            model, argument = cli._setup_ensemble_smoother(args)
            self.assertTrue(isinstance(model, EnsembleSmoother))
            self.assertEquals(3, len(argument.keys()))
            self.assertTrue("active_realizations" in argument)
            self.assertTrue("target_case" in argument)
            self.assertTrue("analysis_module" in argument)

    def test_setup_multiple_data_assimilation(self):
        config_file = self.createTestPath('local/poly_example/poly.ert')
        with ErtTestContext('test_single_test_run', config_file) as work_area:
            ert = work_area.getErt()
            notifier = ErtCliNotifier(ert, config_file)
            ERT.adapt(notifier)
            args = Namespace(realizations="0-4,7,8", weights="6,4,2", target_case="test_case")

            model, argument = cli._setup_multiple_data_assimilation(args)
            self.assertTrue(isinstance(model, MultipleDataAssimilation))
            self.assertEquals(4, len(argument.keys()))
            self.assertTrue("active_realizations" in argument)
            self.assertTrue("target_case" in argument)
            self.assertTrue("analysis_module" in argument)
            self.assertTrue("weights" in argument)

    def test_analysis_module_name_iterable(self):

        active_name = "STD_ENKF"
        modules = ["RML_ENKF"]
        name = cli._get_analysis_module_name(
            active_name, modules, iterable=True)

        self.assertEqual(name, "RML_ENKF")

    def test_analysis_module_name_not_iterable(self):

        active_name = "STD_ENKF"
        modules = ['BOOTSTRAP_ENKF', 'CV_ENKF', 'FWD_STEP_ENKF',
                   'NULL_ENKF', 'SQRT_ENKF', 'STD_ENKF']
        name = cli._get_analysis_module_name(
            active_name, modules, iterable=True)

        self.assertEqual(name, "STD_ENKF")

    def test_analysis_module_name_in_module(self):

        active_name = "STD_ENKF"
        modules = ['STD_ENKF']
        name = cli._get_analysis_module_name(
            active_name, modules, iterable=True)

        self.assertEqual(name, "STD_ENKF")

    def test_analysis_module_items_in_module(self):

        active_name = "FOO"
        modules = ["BAR"]
        name = cli._get_analysis_module_name(
            active_name, modules, iterable=True)

        self.assertEqual(name, "BAR")

    def test_analysis_module_no_hit(self):

        active_name = "FOO"
        modules = []
        name = cli._get_analysis_module_name(
            active_name, modules, iterable=True)

        self.assertIsNone(name)
