import argparse
import sys
import unittest

from ert_gui.main import ert_parser
from argparse import ArgumentParser


class MainTest(unittest.TestCase):

    def test_argparse_exec_gui(self):
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(parser, ['gui', 'test-data/local/poly_example/poly.ert'])
        self.assertEquals(parsed.func.__name__, "runGui")

    def test_argparse_exec_test_run_valid_case(self):
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser, ['test_run', "--verbose", 'test-data/local/poly_example/poly.ert'])
        self.assertEquals(parsed.mode, "test_run")
        self.assertEquals(
            parsed.config, "test-data/local/poly_example/poly.ert")
        self.assertEquals(parsed.func.__name__, "run_cli")
        self.assertTrue(parsed.verbose)

    def test_argparse_exec_ensemble_experiment_valid_case(self):
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(parser, ['ensemble_experiment', "--realizations", "1-4,7,8",
                                     'test-data/local/poly_example/poly.ert'])
        self.assertEquals(parsed.mode, "ensemble_experiment")
        self.assertEquals(
            parsed.config, "test-data/local/poly_example/poly.ert")
        self.assertEquals(parsed.realizations, "1-4,7,8")
        self.assertEquals(parsed.func.__name__, "run_cli")
        self.assertFalse(parsed.verbose)

    def test_argparse_exec_ensemble_experiment_faulty_realizations(self):
        parser = ArgumentParser(prog="test_main")
        with self.assertRaises(SystemExit):
            ert_parser(parser, ['ensemble_experiment', "--realizations", "1~4,7,"
                                'test-data/local/poly_example/poly.ert'])

    def test_argparse_exec_ensemble_smoother_valid_case(self):
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(parser, [
                            'ensemble_smoother', "--target-case", "some_case%d", 'test-data/local/poly_example/poly.ert'])
        self.assertEquals(parsed.mode, "ensemble_smoother")
        self.assertEquals(
            parsed.config, "test-data/local/poly_example/poly.ert")
        self.assertEquals(parsed.target_case, "some_case%d")
        self.assertEquals(parsed.func.__name__, "run_cli")
        self.assertFalse(parsed.verbose)

    def test_argparse_exec_ensemble_smoother_default_target_case(self):
        parser = ArgumentParser(prog="test_main")
        with self.assertRaises(SystemExit):
            ert_parser(parser, ['ensemble_smoother', "--target-case", 'default',
                                'test-data/local/poly_example/poly.ert'])

    def test_argparse_exec_ensemble_smoother_no_target_case(self):
        parser = ArgumentParser(prog="test_main")
        with self.assertRaises(SystemExit):
            ert_parser(parser, ['ensemble_smoother',
                                'test-data/local/poly_example/poly.ert'])

    def test_argparse_exec_es_mda_valid_case(self):
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(parser, ['es_mda', "--target-case", "some_case%d", "--realizations",
                                     "1-10", "--verbose", "--weights", "1, 2, 4", 'test-data/local/poly_example/poly.ert'])
        self.assertEquals(parsed.mode, "es_mda")
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
            parser, ['es_mda', 'test-data/local/poly_example/poly.ert'])
        self.assertEquals(parsed.mode, "es_mda")
        self.assertEquals(
            parsed.config, "test-data/local/poly_example/poly.ert")        
        self.assertEquals(parsed.weights, "3, 2, 1")
        self.assertEquals(parsed.func.__name__, "run_cli")
        self.assertFalse(parsed.verbose)

    def test_argparse_exec_iterated_ensemble_smoother_valid_case(self):
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(parser, ['iterated_ensemble_smoother',  "--iterations", "40",
                                     'test-data/local/poly_example/poly.ert'])
        self.assertEquals(parsed.mode, "iterated_ensemble_smoother")
        self.assertEquals(
            parsed.config, "test-data/local/poly_example/poly.ert")
        self.assertEquals(parsed.iterations, 40)
        self.assertEquals(parsed.func.__name__, "run_cli")
        self.assertFalse(parsed.verbose)

    def test_argparse_exec_iterated_ensemble_smoother_default_iterations(self):
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(parser, [
                            'iterated_ensemble_smoother', 'test-data/local/poly_example/poly.ert'])
        self.assertEquals(parsed.mode, "iterated_ensemble_smoother")
        self.assertEquals(
            parsed.config, "test-data/local/poly_example/poly.ert")
        self.assertEquals(parsed.iterations, 4)
        self.assertEquals(parsed.func.__name__, "run_cli")
        self.assertFalse(parsed.verbose)

    def test_argparse_exec_iterated_ensemble_smoother_faulty_iterations(self):
        parser = ArgumentParser(prog="test_main")
        with self.assertRaises(SystemExit):
            ert_parser(parser, ['iterated_ensemble_smoother', "--iterations", "100"
                                'test-data/local/poly_example/poly.ert'])

        with self.assertRaises(SystemExit):
            ert_parser(parser, ['iterated_ensemble_smoother', "--iterations", "0",
                                'test-data/local/poly_example/poly.ert'])


if __name__ == '__main__':
    unittest.main()
