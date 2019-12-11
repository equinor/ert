import os
import time
import shutil
import sys
import unittest
import datetime

from ecl.util.test import TestAreaContext

from res.simulator import BatchSimulator, BatchContext
from res.job_queue import JobStatusType
from res.enkf import ResConfig
from tests.utils import wait_until, tmpdir
from tests import ResTest
from tests.utils import tmpdir

from threading import Thread

class TestMonitor(object):

    def __init__(self):
        self.sim_context = None

    def start_callback(self, *args, **kwargs): 
        self.sim_context = args[0]

def _wait_for_completion(ctx):
    while ctx.running():
        status = ctx.status
        time.sleep(1)
        sys.stderr.write("status: %s\n" % str(status))
        for job_index in range(len(ctx)):
            status = ctx.job_status(job_index)
            progress = ctx.job_progress(job_index)
            if progress:
                for job in progress.jobs:
                    sys.stderr.write("   %s: \n" % str(job))



class BatchSimulatorTest(ResTest):


    @tmpdir()
    def test_invalid_simulator_creation(self):
        config_file = self.createTestPath("local/batch_sim/batch_sim.ert")

        with TestAreaContext("batch_sim") as test_area:
            test_area.copy_parent_content(config_file)

            # Not valid ResConfig instance as first argument
            with self.assertRaises(ValueError):
                rsim = BatchSimulator("ARG",
                                      {
                                          "WELL_ORDER": ["W1", "W2", "W3"],
                                          "WELL_ON_OFF": ["W1", "W2", "W3"]
                                      },
                                      ["ORDER", "ON_OFF"])

            res_config = ResConfig(user_config_file=os.path.basename(config_file))

            # Control argument not a dict - Exception
            with self.assertRaises(Exception):
                rsim = BatchSimulator(
                    res_config,
                    ["WELL_ORDER", ["W1", "W2", "W3"]],
                    ["ORDER"])

            # Duplicate keys
            with self.assertRaises(ValueError):
                rsim = BatchSimulator(
                    res_config,
                    {"WELL_ORDER": ["W3", "W2", "W3"]},
                    ["ORDER"])

            rsim = BatchSimulator(res_config,
                                  {"WELL_ORDER" : ["W1", "W2", "W3"],
                                   "WELL_ON_OFF" : ["W1", "W2", "W3"]},
                                  ["ORDER", "ON_OFF"])

            # The key for one of the controls is invalid => KeyError
            with self.assertRaises(KeyError):
                rsim.start("case",
                           [
                               (2,
                                {
                                    "WELL_ORDERX": {"W1": 0, "W2": 0, "W3": 1},
                                    "WELL_ON_OFF": {"W1": 0, "W2": 0, "W3": 1},
                                }),
                               (2,
                                {
                                    "WELL_ORDER": {"W1": 0, "W2": 0, "W3": 0},
                                    "WELL_ON_OFF": {"W1": 0, "W2": 0, "W3": 1},
                                }),
                           ])

            # The key for one of the variables is invalid => KeyError
            with self.assertRaises(KeyError):
                rsim.start("case",
                           [
                               (2,
                                {
                                    "WELL_ORDER": {"W1": 0, "W4": 0, "W3": 1},
                                    "WELL_ON_OFF": {"W1": 0, "W2": 0, "W3": 1},
                                }),
                               (1,
                                {
                                    "WELL_ORDER": {"W1": 0, "W2": 0, "W3": 0},
                                    "WELL_ON_OFF": {"W1": 0, "W2": 0, "W3": 1},
                                }),
                           ])

            # The key for one of the variables is invalid => KeyError
            with self.assertRaises(KeyError):
                rsim.start("case",
                           [
                               (2,
                                {
                                    "WELL_ORDER": {"W1": 0, "W2": 0, "W3": 1, "W0": 0},
                                    "WELL_ON_OFF": {"W1": 0, "W2": 0, "W3": 1},
                                }),
                               (1,
                                {
                                    "WELL_ORDER": {"W1": 0, "W2": 0, "W3": 0},
                                    "WELL_ON_OFF": {"W1": 0, "W2": 0, "W3": 1},
                                }),
                           ])


            # Missing the key WELL_ON_OFF => KeyError
            with self.assertRaises(KeyError):
                rsim.start("case", [
                    (2, {"WELL_ORDER" : {"W1": 0, "W2": 0, "W3": 1}})])

            # One of the numeric vectors has wrong length => ValueError:
            with self.assertRaises(KeyError):
                rsim.start("case",
                           [
                               (2,
                                {
                                    "WELL_ORDER": {"W1": 0, "W2": 0, "W3": 1},
                                    "WELL_ON_OFF": {"W2": 0}
                                }),
                           ])

            # Not numeric values => Exception
            with self.assertRaises(Exception):
                rsim.start("case",
                           [
                               (2,
                                {
                                    "WELL_ORDER": {"W1": 0, "W2": 0, "W3": 1},
                                    "WELL_ON_OFF": {"W1": 0, "W2": 1, "W3": 'X'}
                                }),
                           ])

            # Not numeric values => Exception
            with self.assertRaises(Exception):
                rsim.start("case",
                           [
                               ('2',
                                {
                                    "WELL_ORDER": {"W1": 0, "W2": 0, "W3": 1},
                                    "WELL_ON_OFF" : {"W1": 0, "W2": 1, "W3": 4},
                                }),
                           ])


    @tmpdir()
    def test_batch_simulation(self):
        config_file = self.createTestPath("local/batch_sim/batch_sim.ert")

        with TestAreaContext("batch_sim") as test_area:
            test_area.copy_parent_content(config_file)

            res_config = ResConfig(user_config_file=os.path.basename(config_file))
            monitor = TestMonitor()
            rsim = BatchSimulator(res_config,
                                  {
                                      "WELL_ORDER" : ["W1", "W2", "W3"],
                                      "WELL_ON_OFF" : ["W1", "W2", "W3"]
                                  },
                                  ["ORDER", "ON_OFF"],
                                  callback=monitor.start_callback)

            # Starting a simulation which should actually run through.
            case_data = [
                (2,
                 {
                     "WELL_ORDER": {"W1": 1, "W2": 2, "W3": 3},
                     "WELL_ON_OFF": {"W1": 4, "W2": 5, "W3": 6}
                 }),
                (1,
                 {
                     "WELL_ORDER": {"W1": 7, "W2": 8, "W3": 9},
                     "WELL_ON_OFF" : {"W1": 10, "W2": 11, "W3": 12}
                 }),
            ]

            ctx = rsim.start("case", case_data)
            self.assertEqual(len(case_data), len(ctx))

            # Asking for results before it is complete.
            with self.assertRaises(RuntimeError):
                ctx.results()

            # Ask for status of simulation we do not have.
            with self.assertRaises(KeyError):
                ctx.job_status(1973)

            with self.assertRaises(KeyError):
                ctx.job_progress(1987)

            # Carry out simulations..
            _wait_for_completion(ctx)

            # Fetch and validate results
            results = ctx.results()
            self.assertEqual(len(results), 2)

            for result, (_, controls) in zip(results, case_data):
                self.assertEqual(sorted(["ORDER", "ON_OFF"]),
                                 sorted(result.keys()))

                for res_key, ctrl_key in (
                        ("ORDER", "WELL_ORDER"),
                        ("ON_OFF", "WELL_ON_OFF"),
                    ):

                    # The forward model job SQUARE_PARAMS will load the control
                    # values and square them before writing results to disk in
                    # the order W1, W2, W3.
                    self.assertEqual(
                        [controls[ctrl_key][var_name] ** 2 for var_name in ["W1", "W2", "W3"]],
                        list(result[res_key])
                        )

            self.assertTrue(isinstance(monitor.sim_context, BatchContext))

    @tmpdir()
    def test_batch_simulation_invalid_suffixes(self):
        config_file = self.createTestPath("local/batch_sim/batch_sim.ert")
        with TestAreaContext("batch_sim") as test_area:
            test_area.copy_parent_content(config_file)
            res_config = ResConfig(user_config_file=os.path.basename(config_file))

            # If suffixes are given, must be all non-empty string collections
            type_err_suffixes = (
                27,
                "astring",
                b"somebytes",
                True,
                False,
                [True, False],
                None,
                range(3),
                )
            for sfx in type_err_suffixes:
                with self.assertRaises(TypeError):
                    BatchSimulator(res_config, {
                        "WELL_ORDER" : { "W1" : ["a"], "W3" : sfx },
                        }, ["ORDER"])
            val_err_suffixes = (
                [],
                {},
                [""],
                ["a", "a"],
                )
            for sfx in val_err_suffixes:
                with self.assertRaises(ValueError):
                    BatchSimulator(res_config, {
                        "WELL_ORDER" : { "W1" : ["a"], "W3" : sfx },
                        }, ["ORDER"])

            rsim = BatchSimulator(res_config, {
                "WELL_ORDER" : {
                    "W1" : ["a", "b"],
                    "W3" : ["c"],
                    },
                },
                ["ORDER"])

            # suffixes not taken into account
            with self.assertRaises(KeyError):
                rsim.start("case",
                           [(1, {"WELL_ORDER": { "W1": 3, "W3": 2 }})])
            with self.assertRaises(KeyError):
                rsim.start("case",
                           [(1, {"WELL_ORDER": { "W1": {}, "W3": {} }})])

            # wrong suffixes
            with self.assertRaises(KeyError):
                rsim.start("case", [(1, {"WELL_ORDER": {
                    "W1": { "a": 3, "x": 3 },
                    "W3": { "c": 2 },
                    }})])

            # missing one suffix
            with self.assertRaises(KeyError):
                rsim.start("case", [(1, {"WELL_ORDER": {
                    "W1": { "a": 3 },
                    "W3": { "c": 2 },
                    }})])

            # wrong type for values
            # Exception cause atm this would raise a ctypes.ArgumentError
            # but that's an implementation detail that will hopefully change
            # not so far in the future
            with self.assertRaises(Exception):
                rsim.start("case", [(1, {"WELL_ORDER": {
                    "W1": { "a": "3", "b": 3 },
                    "W3": { "c": 2 },
                    }})])


    @tmpdir()
    def test_batch_simulation_suffixes(self):
        config_file = self.createTestPath("local/batch_sim/batch_sim.ert")
        with TestAreaContext("batch_sim") as test_area:
            test_area.copy_parent_content(config_file)

            res_config = ResConfig(user_config_file=os.path.basename(config_file))
            monitor = TestMonitor()
            rsim = BatchSimulator(res_config,
                                  {
                                      "WELL_ORDER" : {
                                          "W1" : ["a", "b"],
                                          "W2" : ["c"],
                                          "W3" : ["a", "b"],
                                          },
                                      "WELL_ON_OFF" : ["W1", "W2", "W3"]
                                  },
                                  ["ORDER", "ON_OFF"],
                                  callback=monitor.start_callback)
            # Starting a simulation which should actually run through.
            case_data = [
                (2, {
                    "WELL_ORDER": {
                        "W1": {"a": 0.5, "b": 0.2},
                        "W2": {"c": 2},
                        "W3": {"a":-0.5, "b":-0.2},
                        },
                    "WELL_ON_OFF": {"W1": 4, "W2": 5, "W3": 6}
                }),
                (1, {
                    "WELL_ORDER": {
                        "W1": {"a": 0.8, "b": 0.9},
                        "W2": {"c": 1.6},
                        "W3": {"a":-0.8, "b":-0.9},
                        },
                    "WELL_ON_OFF" : {"W1": 10, "W2": 11, "W3": 12}
                }),
            ]

            ctx = rsim.start("case", case_data)
            self.assertEqual(len(case_data), len(ctx))
            _wait_for_completion(ctx)

            # Fetch and validate results
            results = ctx.results()
            self.assertEqual(len(results), 2)

            for result in results:
                self.assertEqual(sorted(["ORDER", "ON_OFF"]),
                                 sorted(result.keys()))

            keys = ("W1", "W2", "W3")
            for result, (_, controls) in zip(results, case_data):
                expected = [controls["WELL_ON_OFF"][key] ** 2 for key in keys]
                self.assertEqual(expected, list(result["ON_OFF"]))

                expected = [v ** 2
                            for key in keys
                            for _, v in controls["WELL_ORDER"][key].items()]
                for exp, act in zip(expected, list(result["ORDER"])):
                    self.assertAlmostEqual(exp, act)


    @tmpdir()
    def test_stop_sim(self):
        config_file = self.createTestPath("local/batch_sim/batch_sim.ert")
        with TestAreaContext("batch_sim_stop") as test_area:
            test_area.copy_parent_content(config_file)
            res_config = ResConfig(user_config_file=os.path.basename(config_file))

            rsim = BatchSimulator(res_config,
                                  {
                                      "WELL_ORDER" : ["W1", "W2", "W3"],
                                      "WELL_ON_OFF" : ["W1", "W2", "W3"]
                                  },
                                  ["ORDER", "ON_OFF"])

            case_name = 'MyCaseName_123'

            # Starting a simulation which should actually run through.
            ctx = rsim.start(case_name,
                             [
                                 (2,
                                  {
                                      "WELL_ORDER": {"W1": 1, "W2": 2, "W3": 3},
                                      "WELL_ON_OFF": {"W1": 4, "W2": 5, "W3": 6}
                                  }),
                                 (1,
                                  {
                                      "WELL_ORDER": {"W1": 7, "W2": 8, "W3": 9},
                                      "WELL_ON_OFF": {"W1": 10, "W2": 11, "W3": 12}
                                  })
                             ])

            ctx.stop()
            status = ctx.status

            self.assertEqual(status.complete, 0)
            self.assertEqual(status.running, 0)

            runpath = 'storage/batch_sim/runpath/%s/realisation-0' % case_name
            self.assertTrue(os.path.exists(runpath))

    def assertContextStatusOddFailures(self, batch_ctx, final_state_only=False):
        running_status = set((
            JobStatusType.JOB_QUEUE_WAITING,
            JobStatusType.JOB_QUEUE_SUBMITTED,
            JobStatusType.JOB_QUEUE_PENDING,
            JobStatusType.JOB_QUEUE_RUNNING,
            JobStatusType.JOB_QUEUE_UNKNOWN,
            JobStatusType.JOB_QUEUE_EXIT,
            JobStatusType.JOB_QUEUE_DONE,
        ))

        for idx in range(len(batch_ctx)):
            status = batch_ctx.job_status(idx)
            if not final_state_only and status in running_status:
                continue
            elif idx%2 == 0:
                self.assertEqual(JobStatusType.JOB_QUEUE_SUCCESS, status)
            else:
                self.assertEqual(JobStatusType.JOB_QUEUE_FAILED, status)

    @tmpdir()
    def test_batch_ctx_status_failing_jobs(self):

        config_dir = os.path.join(self.SOURCE_ROOT, "test-data/local/batch_sim")
        shutil.copytree(config_dir, "batch_sim")
        os.chdir("batch_sim")

        self.assertTrue(os.path.isfile("batch_sim_sleep_and_fail.ert"))
        res_config = ResConfig(user_config_file="batch_sim_sleep_and_fail.ert")

        external_parameters = {
            "WELL_ORDER": ("W1", "W2", "W3"),
            "WELL_ON_OFF": ("W1", "W2", "W3"),
        }
        results = ("ORDER", "ON_OFF")
        rsim = BatchSimulator(res_config, external_parameters, results)

        cases = [
            (
                0,
                {
                    "WELL_ORDER": {"W1": idx+1, "W2": idx+2, "W3": idx+3},
                    "WELL_ON_OFF": {"W1": idx*4, "W2": idx*5, "W3": idx*6}
                }
            )
            for idx in range(10)
        ]

        batch_ctx = rsim.start("case_name", cases)
        while batch_ctx.running():
            self.assertContextStatusOddFailures(batch_ctx)
            time.sleep(1)

        self.assertContextStatusOddFailures(batch_ctx, final_state_only=True)


if __name__ == "__main__":
    unittest.main()
