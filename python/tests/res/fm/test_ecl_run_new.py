#  Copyright (C) 2018  Equinor ASA, Norway.
#
#  The file 'test_ecl.py' is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.
import os
import stat
import unittest
import shutil
import functools
import inspect
import yaml
import pytest
import re

from ecl.summary import EclSum
from ecl.util.test import TestAreaContext
from tests import ResTest
from tests.utils import tmpdir
from res.fm.ecl import run, Ecl100Config, EclrunConfig, EclRun
from subprocess import Popen, PIPE
from subprocess import Popen, PIPE
from distutils.spawn import find_executable
from _pytest.monkeypatch import MonkeyPatch


def find_version(output):
    return re.search(r"flow\s*([\d.]+)", output).group(1)


class EclRunTest(ResTest):
    def setUp(self):
        self.ecl_config_path = os.path.dirname(inspect.getsourcefile(Ecl100Config))
        self.monkeypatch = MonkeyPatch()

    def tearDown(self):
        self.monkeypatch.undo()

    @staticmethod
    def _eclrun_conf():
        return {
            "eclrun_env": {
                "SLBSLS_LICENSE_FILE": "7321@eclipse-lic-no.statoil.no",
                "ECLPATH": "/prog/res/ecl/grid",
                "PATH": "/prog/res/ecl/grid/macros",
                "F_UFMTENDIAN": "big",
            }
        }

    def init_eclrun_config(self):
        conf = EclRunTest._eclrun_conf()
        with open("ecl100_config.yml", "w") as f:
            f.write(yaml.dump(conf))
        self.monkeypatch.setenv("ECL100_SITE_CONFIG", "ecl100_config.yml")

    @tmpdir()
    @pytest.mark.equinor_test
    def test_run(self):
        self.init_eclrun_config()
        shutil.copy(
            os.path.join(self.SOURCE_ROOT, "test-data/local/eclipse/SPE1.DATA"),
            "SPE1.DATA",
        )
        ecl_config = Ecl100Config()

        ecl_run = EclRun("SPE1.DATA", None)
        ecl_run.runEclipse(eclrun_config=EclrunConfig(ecl_config, "2019.3"))

        ok_path = os.path.join(ecl_run.runPath(), "{}.OK".format(ecl_run.baseName()))
        log_path = os.path.join(ecl_run.runPath(), "{}.LOG".format(ecl_run.baseName()))

        self.assertTrue(os.path.isfile(ok_path))
        self.assertTrue(os.path.isfile(log_path))
        self.assertTrue(os.path.getsize(log_path) > 0)

        errors = ecl_run.parseErrors()
        self.assertEqual(0, len(errors))

    @pytest.mark.equinor_test
    @tmpdir()
    def test_run_api(self):
        self.init_eclrun_config()
        shutil.copy(
            os.path.join(self.SOURCE_ROOT, "test-data/local/eclipse/SPE1.DATA"),
            "SPE1.DATA",
        )
        ecl_config = Ecl100Config()
        run(ecl_config, ["SPE1.DATA", "--version=2019.3"])

        self.assertTrue(os.path.isfile("SPE1.DATA"))

    @pytest.mark.equinor_test
    @tmpdir()
    def test_failed_run(self):
        self.init_eclrun_config()
        shutil.copy(
            os.path.join(self.SOURCE_ROOT, "test-data/local/eclipse/SPE1_ERROR.DATA"),
            "SPE1_ERROR.DATA",
        )
        ecl_config = Ecl100Config()
        eclrun_config = EclrunConfig(ecl_config, "2019.3")
        ecl_run = EclRun("SPE1_ERROR", None)
        with self.assertRaises(Exception) as error_context:
            ecl_run.runEclipse(eclrun_config=eclrun_config)
        self.assertIn("ERROR", str(error_context.exception))

    @pytest.mark.equinor_test
    @tmpdir()
    def test_failed_run_OK(self):
        self.init_eclrun_config()
        shutil.copy(
            os.path.join(self.SOURCE_ROOT, "test-data/local/eclipse/SPE1_ERROR.DATA"),
            "SPE1_ERROR.DATA",
        )
        ecl_config = Ecl100Config()
        run(ecl_config, ["SPE1_ERROR", "--version=2019.3", "--ignore-errors"])

    @pytest.mark.equinor_test
    @tmpdir()
    def test_mpi_run(self):
        self.init_eclrun_config()
        shutil.copy(
            os.path.join(
                self.SOURCE_ROOT, "test-data/local/eclipse/SPE1_PARALLELL.DATA"
            ),
            "SPE1_PARALLELL.DATA",
        )
        ecl_config = Ecl100Config()
        run(ecl_config, ["SPE1_PARALLELL.DATA", "--version=2019.3", "--num-cpu=2"])
        self.assertTrue(os.path.isfile("SPE1_PARALLELL.LOG"))
        self.assertTrue(os.path.getsize("SPE1_PARALLELL.LOG") > 0)

    @pytest.mark.equinor_test
    @tmpdir()
    def test_summary_block(self):
        self.init_eclrun_config()
        shutil.copy(
            os.path.join(self.SOURCE_ROOT, "test-data/local/eclipse/SPE1.DATA"),
            "SPE1.DATA",
        )
        ecl_config = Ecl100Config()
        ecl_run = EclRun("SPE1.DATA", None)
        ret_value = ecl_run.summary_block()
        self.assertTrue(ret_value is None)

        ecl_run.runEclipse(eclrun_config=EclrunConfig(ecl_config, "2019.3"))
        ecl_sum = ecl_run.summary_block()
        self.assertTrue(isinstance(ecl_sum, EclSum))

    @pytest.mark.equinor_test
    @tmpdir()
    def test_check(self):
        full_case = os.path.join(
            self.SOURCE_ROOT, "test-data/Equinor/ECLIPSE/Gurbat/ECLIPSE"
        )
        short_case = os.path.join(
            self.SOURCE_ROOT, "test-data/Equinor/ECLIPSE/ShortSummary/ECLIPSE"
        )
        failed_case = os.path.join(
            self.SOURCE_ROOT,
            "test-data/Equinor/ECLIPSE/SummaryFail/NOR-2013A_R002_1208-0",
        )

        with self.assertRaises(IOError):
            self.assertTrue(EclRun.checkCase(full_case, failed_case))

        with self.assertRaises(IOError):
            self.assertTrue(EclRun.checkCase(full_case, "DOES-NOT-EXIST"))

        with self.assertRaises(IOError):
            self.assertTrue(EclRun.checkCase("DOES-NOT-EXIST", full_case))

        with self.assertRaises(ValueError):
            EclRun.checkCase(full_case, short_case)

        self.assertTrue(not os.path.isfile("CHECK_ECLIPSE_RUN.OK"))
        self.assertTrue(EclRun.checkCase(full_case, full_case))
        self.assertTrue(os.path.isfile("CHECK_ECLIPSE_RUN.OK"))

        os.remove("CHECK_ECLIPSE_RUN.OK")
        self.assertTrue(
            EclRun.checkCase(short_case, full_case)
        )  # Simulation is longer than refcase - OK
        self.assertTrue(os.path.isfile("CHECK_ECLIPSE_RUN.OK"))
