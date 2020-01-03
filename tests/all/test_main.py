import argparse
import sys
import unittest

from ert_shared.main import ert_parser
from argparse import ArgumentParser
from ert_shared.cli import (
    ENSEMBLE_SMOOTHER_MODE,
    ENSEMBLE_EXPERIMENT_MODE,
    ES_MDA_MODE,
    TEST_RUN_MODE,
    WORKFLOW_MODE
)


class MainTest(unittest.TestCase):

    def test_argparse_exec_gui(self):
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(parser, ['gui', 'test-data/local/poly_example/poly.ert'])
        self.assertEquals(parsed.func.__name__, "run_gui_wrapper")

    def test_argparse_exec_test_run_valid_case(self):
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser, [TEST_RUN_MODE, "--verbose", 'test-data/local/poly_example/poly.ert'])
        self.assertEquals(parsed.mode, TEST_RUN_MODE)
        self.assertEquals(
            parsed.config, "test-data/local/poly_example/poly.ert")
        self.assertEquals(parsed.func.__name__, "run_cli")
        self.assertTrue(parsed.verbose)

    def test_argparse_exec_ensemble_experiment_valid_case(self):
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(parser, [ENSEMBLE_EXPERIMENT_MODE, "--realizations", "1-4,7,8",
                                     'test-data/local/poly_example/poly.ert'])
        self.assertEquals(parsed.mode, ENSEMBLE_EXPERIMENT_MODE)
        self.assertEquals(
            parsed.config, "test-data/local/poly_example/poly.ert")
        self.assertEquals(parsed.realizations, "1-4,7,8")
        self.assertEquals(parsed.func.__name__, "run_cli")
        self.assertFalse(parsed.verbose)

    def test_argparse_exec_ensemble_experiment_faulty_realizations(self):
        parser = ArgumentParser(prog="test_main")
        with self.assertRaises(SystemExit):
            ert_parser(parser, [ENSEMBLE_EXPERIMENT_MODE, "--realizations", "1~4,7,"
                                'test-data/local/poly_example/poly.ert'])

    def test_argparse_exec_ensemble_smoother_valid_case(self):
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(parser, [
                            ENSEMBLE_SMOOTHER_MODE, "--target-case", "some_case", 'test-data/local/poly_example/poly.ert'])
        self.assertEquals(parsed.mode, ENSEMBLE_SMOOTHER_MODE)
        self.assertEquals(
            parsed.config, "test-data/local/poly_example/poly.ert")
        self.assertEquals(parsed.target_case, "some_case")
        self.assertEquals(parsed.func.__name__, "run_cli")
        self.assertFalse(parsed.verbose)


    def test_argparse_exec_ensemble_smoother_no_target_case(self):
        parser = ArgumentParser(prog="test_main")
        with self.assertRaises(SystemExit):
            ert_parser(parser, [ENSEMBLE_SMOOTHER_MODE,
                                'test-data/local/poly_example/poly.ert'])

    def test_argparse_exec_es_mda_valid_case(self):
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(parser, [ES_MDA_MODE, "--target-case", "some_case%d", "--realizations",
                                     "1-10", "--verbose", "--weights", "1, 2, 4", 'test-data/local/poly_example/poly.ert'])
        self.assertEquals(parsed.mode, ES_MDA_MODE)
        self.assertEquals(
            parsed.config, "test-data/local/poly_example/poly.ert")
        self.assertEquals(parsed.target_case, "some_case%d")
        self.assertEquals(parsed.realizations, "1-10")        
        self.assertEquals(parsed.weights, "1, 2, 4")
        self.assertEquals(parsed.func.__name__, "run_cli")
        self.assertTrue(parsed.verbose)

    def test_argparse_exec_es_mda_default_weights(self):
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser, [ES_MDA_MODE, 'test-data/local/poly_example/poly.ert'])
        self.assertEquals(parsed.mode, ES_MDA_MODE)
        self.assertEquals(
            parsed.config, "test-data/local/poly_example/poly.ert")        
        self.assertEquals(parsed.weights, "4, 2, 1")
        self.assertEquals(parsed.func.__name__, "run_cli")
        self.assertFalse(parsed.verbose)

    def test_argparse_exec_workflow(self):
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser, [WORKFLOW_MODE, "--verbose", "workflow_name", 'test-data/local/poly_example/poly.ert'])
        self.assertEquals(parsed.mode, WORKFLOW_MODE)
        self.assertEquals(parsed.name, "workflow_name")
        self.assertEquals(
            parsed.config, "test-data/local/poly_example/poly.ert")
        self.assertEquals(parsed.func.__name__, "run_cli")
        self.assertTrue(parsed.verbose)

    def test_argparse_exec_ensemble_experiment_current_case(self):
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(parser, [ENSEMBLE_EXPERIMENT_MODE, "--current-case", 'test_case',
                                     'test-data/local/poly_example/poly.ert'])
        self.assertEquals(parsed.mode, ENSEMBLE_EXPERIMENT_MODE)
        self.assertEquals(
            parsed.config, "test-data/local/poly_example/poly.ert")
        self.assertEquals(parsed.current_case, 'test_case')
        self.assertEquals(parsed.func.__name__, "run_cli")
        self.assertFalse(parsed.verbose)
    
    def test_argparse_exec_ensemble_smoother_current_case(self):
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(parser, [ENSEMBLE_SMOOTHER_MODE, "--current-case", 'test_case',
                                     "--target-case", 'test_case_smoother',
                                     'test-data/local/poly_example/poly.ert'])
        self.assertEquals(parsed.mode, ENSEMBLE_SMOOTHER_MODE)
        self.assertEquals(
            parsed.config, "test-data/local/poly_example/poly.ert")
        self.assertEquals(parsed.current_case, 'test_case')
        self.assertEquals(parsed.func.__name__, "run_cli")
        self.assertFalse(parsed.verbose)
    
    def test_argparse_exec_ensemble_es_mda_current_case(self):
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(parser, [ES_MDA_MODE, "--current-case", 'test_case',
                                     'test-data/local/poly_example/poly.ert'])
        self.assertEquals(parsed.mode, ES_MDA_MODE)
        self.assertEquals(
            parsed.config, "test-data/local/poly_example/poly.ert")
        self.assertEquals(parsed.current_case, 'test_case')
        self.assertEquals(parsed.func.__name__, "run_cli")
        self.assertFalse(parsed.verbose)


if __name__ == '__main__':
    unittest.main()
