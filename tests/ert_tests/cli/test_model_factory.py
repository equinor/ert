import os
from argparse import Namespace

from _pytest.tmpdir import tmp_path
from ecl.util.util import BoolVector
from ert_utils import ErtTest

import ert_shared.cli.model_factory as model_factory
from ert_shared import ERT
from ert_shared.cli.notifier import ErtCliNotifier
from ert_shared.models.ensemble_experiment import EnsembleExperiment
from ert_shared.models.ensemble_smoother import EnsembleSmoother
from ert_shared.models.multiple_data_assimilation import MultipleDataAssimilation
from ert_shared.models.iterated_ensemble_smoother import IteratedEnsembleSmoother
from ert_shared.models.single_test_run import SingleTestRun
from res.test import ErtTestContext


class ModelFactoryTest(ErtTest):
    def test_custom_target_case_name(self):
        config_file = self.createTestPath("local/poly_example/poly.ert")
        with ErtTestContext("test_custom_target_case_name", config_file) as work_area:
            ert = work_area.getErt()
            notifier = ErtCliNotifier(ert, config_file)
            with ERT.adapt(notifier):
                custom_name = "test"
                args = Namespace(target_case=custom_name)
                res = model_factory._target_case_name(ert, args)
                self.assertEqual(custom_name, res)

    def test_default_target_case_name(self):
        config_file = self.createTestPath("local/poly_example/poly.ert")
        with ErtTestContext("test_default_target_case_name", config_file) as work_area:
            ert = work_area.getErt()
            notifier = ErtCliNotifier(ert, config_file)

            with ERT.adapt(notifier):

                args = Namespace(target_case=None)
                res = model_factory._target_case_name(ert, args)
                self.assertEqual("default_smoother_update", res)

    def test_default_target_case_name_format_mode(self):
        config_file = self.createTestPath("local/poly_example/poly.ert")
        with ErtTestContext(
            "test_default_target_case_name_format_mode", config_file
        ) as work_area:
            ert = work_area.getErt()
            notifier = ErtCliNotifier(ert, config_file)
            with ERT.adapt(notifier):

                args = Namespace(target_case=None)
                res = model_factory._target_case_name(ert, args, format_mode=True)
                self.assertEqual("default_%d", res)

    def test_default_realizations(self):
        config_file = self.createTestPath("local/poly_example/poly.ert")
        with ErtTestContext("test_default_realizations", config_file) as work_area:
            ert = work_area.getErt()
            notifier = ErtCliNotifier(ert, config_file)
            with ERT.adapt(notifier):

                args = Namespace(realizations=None)
                res = model_factory._realizations(args)
                ensemble_size = ERT.ert.getEnsembleSize()
                mask = BoolVector(default_value=False, initial_size=ensemble_size)
                mask.updateActiveMask("0-99")
                self.assertEqual(mask, res)

    def test_init_iteration_number(self):
        config_file = self.createTestPath("local/poly_example/poly.ert")
        with ErtTestContext("test_init_iteration_number", config_file) as work_area:
            ert = work_area.getErt()
            notifier = ErtCliNotifier(ert, config_file)
            with ERT.adapt(notifier):
                args = Namespace(iter_num=10, realizations=None)
                model, argument = model_factory._setup_ensemble_experiment(ert, args)
                run_context = model.create_context(argument)
                self.assertEqual(argument["iter_num"], 10)
                self.assertEqual(run_context.get_iter(), 10)

    def test_custom_realizations(self):
        config_file = self.createTestPath("local/poly_example/poly.ert")
        with ErtTestContext("test_custom_realizations", config_file) as work_area:
            ert = work_area.getErt()
            notifier = ErtCliNotifier(ert, config_file)
            with ERT.adapt(notifier):

                args = Namespace(realizations="0-4,7,8")
                res = model_factory._realizations(args)
                ensemble_size = ERT.ert.getEnsembleSize()
                mask = BoolVector(default_value=False, initial_size=ensemble_size)
                mask.updateActiveMask("0-4,7,8")
                self.assertEqual(mask, res)

    def test_setup_single_test_run(self):
        config_file = self.createTestPath("local/poly_example/poly.ert")
        with ErtTestContext("test_single_test_run", config_file) as work_area:
            ert = work_area.getErt()
            notifier = ErtCliNotifier(ert, config_file)
            with ERT.adapt(notifier):

                model, argument = model_factory._setup_single_test_run(ert)
                self.assertTrue(isinstance(model, SingleTestRun))
                self.assertEqual(1, len(argument.keys()))
                self.assertTrue("active_realizations" in argument)
                model.create_context(argument)

    def test_setup_ensemble_experiment(self):
        config_file = self.createTestPath("local/poly_example/poly.ert")
        with ErtTestContext("test_single_test_run", config_file) as work_area:
            ert = work_area.getErt()
            notifier = ErtCliNotifier(ert, config_file)
            with ERT.adapt(notifier):

                model, argument = model_factory._setup_single_test_run(ert)
                self.assertTrue(isinstance(model, EnsembleExperiment))
                self.assertEqual(1, len(argument.keys()))
                self.assertTrue("active_realizations" in argument)
                model.create_context(argument)

    def test_setup_ensemble_smoother(self):
        config_file = self.createTestPath("local/poly_example/poly.ert")
        with ErtTestContext("test_single_test_run", config_file) as work_area:
            ert = work_area.getErt()
            notifier = ErtCliNotifier(ert, config_file)
            with ERT.adapt(notifier):
                args = Namespace(realizations="0-4,7,8", target_case="test_case")

                model, argument = model_factory._setup_ensemble_smoother(ert, args)
                self.assertTrue(isinstance(model, EnsembleSmoother))
                self.assertEqual(3, len(argument.keys()))
                self.assertTrue("active_realizations" in argument)
                self.assertTrue("target_case" in argument)
                self.assertTrue("analysis_module" in argument)
                model.create_context(argument)

    def test_setup_multiple_data_assimilation(self):
        config_file = self.createTestPath("local/poly_example/poly.ert")
        with ErtTestContext("test_single_test_run", config_file) as work_area:
            ert = work_area.getErt()
            notifier = ErtCliNotifier(ert, config_file)
            with ERT.adapt(notifier):
                args = Namespace(
                    realizations="0-4,7,8",
                    weights="6,4,2",
                    target_case="test_case_%d",
                    start_iteration="0",
                )

                model, argument = model_factory._setup_multiple_data_assimilation(
                    ert, args
                )
                self.assertTrue(isinstance(model, MultipleDataAssimilation))
                self.assertEqual(5, len(argument.keys()))
                self.assertTrue("active_realizations" in argument)
                self.assertTrue("target_case" in argument)
                self.assertTrue("analysis_module" in argument)
                self.assertTrue("weights" in argument)
                self.assertTrue("start_iteration" in argument)
                model.create_context(argument, 0)

    def test_setup_iterative_ensemble_smoother(self):
        config_file = self.createTestPath("local/poly_example/poly.ert")
        with ErtTestContext(
            "_setup_iterative_ensemble_smoother", config_file
        ) as work_area:
            ert = work_area.getErt()
            notifier = ErtCliNotifier(ert, config_file)
            with ERT.adapt(notifier):
                args = Namespace(
                    realizations="0-4,7,8",
                    target_case="test_case_%d",
                    num_iterations="10",
                )

                model, argument = model_factory._setup_iterative_ensemble_smoother(
                    ert, args
                )
                self.assertTrue(isinstance(model, IteratedEnsembleSmoother))
                self.assertEqual(4, len(argument.keys()))
                self.assertTrue("active_realizations" in argument)
                self.assertTrue("target_case" in argument)
                self.assertTrue("analysis_module" in argument)
                self.assertTrue("num_iterations" in argument)
                self.assertTrue(ERT.enkf_facade.get_number_of_iterations() == 10)
                model.create_context(argument, 0)

    def test_analysis_module_name_iterable(self):

        active_name = "STD_ENKF"
        modules = ["LIB_IES"]
        name = model_factory._get_analysis_module_name(
            active_name, modules, iterable=True
        )

        self.assertEqual(name, "LIB_IES")

    def test_analysis_module_name_not_iterable(self):

        active_name = "STD_ENKF"
        modules = [
            "STD_ENKF",
        ]
        name = model_factory._get_analysis_module_name(
            active_name, modules, iterable=True
        )

        self.assertEqual(name, "STD_ENKF")

    def test_analysis_module_name_in_module(self):

        active_name = "STD_ENKF"
        modules = ["STD_ENKF"]
        name = model_factory._get_analysis_module_name(
            active_name, modules, iterable=True
        )

        self.assertEqual(name, "STD_ENKF")

    def test_analysis_module_items_in_module(self):

        active_name = "FOO"
        modules = ["BAR"]
        name = model_factory._get_analysis_module_name(
            active_name, modules, iterable=True
        )

        self.assertEqual(name, "BAR")

    def test_analysis_module_no_hit(self):

        active_name = "FOO"
        modules = []
        name = model_factory._get_analysis_module_name(
            active_name, modules, iterable=True
        )

        self.assertIsNone(name)
