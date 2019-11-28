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
from res.fm.ecl import *
from subprocess import Popen, PIPE
from distutils.spawn import find_executable
from _pytest.monkeypatch import MonkeyPatch


def flow_install():
    try:
        Popen(["flow"])
        return True
    except OSError:
        pass
    return False


flow_installed = pytest.mark.skipif(not flow_install(), reason="Requires flow")


def find_version(output):
    return re.search(r'flow\s*([\d.]+)', output).group(1)


class EclRunTest(ResTest):
    def setUp(self):
        self.ecl_config_path = os.path.dirname( inspect.getsourcefile(Ecl100Config) )
        self.monkeypatch = MonkeyPatch()


    def tearDown(self):
        self.monkeypatch.undo()


    def init_ecl100_config(self):
        conf = {"env" : {"F_UFMTENDIAN" : "big",
                         "LM_LICENSE_FILE" : "7321@eclipse-lic-no.statoil.no",
                         "ARCH" : "x86_64"},
                "versions" : {"2014.2" : {"scalar": {"executable" : "/prog/ecl/grid/2014.2/bin/linux_x86_64/eclipse.exe"},
                                          "mpi"   : {"executable" : "/prog/ecl/grid/2014.2/bin/linux_x86_64/eclipse_ilmpi.exe",
                                                     "mpirun" : "/prog/ecl/grid/tools/linux_x86_64/intel/mpi/5.0.2.044/bin64/mpirun",
                                                     "env" : {"I_MPI_ROOT" : "/prog/ecl/grid/tools/linux_x86_64/intel/mpi/5.0.2.044/",
                                                              "P4_RSHCOMMAND" : "ssh",
                                                              "LD_LIBRARY_PATH" : "/prog/ecl/grid/tools/linux_x86_64/intel/mpi/5.0.2.044/lib64:$LD_LIBRARY_PATH",
                                                              "PATH" : "/prog/ecl/grid/tools/linux_x86_64/intel/mpi/5.0.2.044/bin64:$PATH"}}}}}
        with open("ecl100_config.yml","w") as f:
            f.write( yaml.dump(conf) )
        self.monkeypatch.setenv("ECL100_SITE_CONFIG", "ecl100_config.yml")


    def init_flow_config(self):
        version = "2018.10"

        p = Popen(["flow", '--version'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        output, err = p.communicate()
        rc = p.returncode
        if rc == 0:
            version = find_version(output)
        path_to_exe = find_executable("flow")

        conf = {
            "default_version": version,
            "versions": {
                version: {
                    "scalar": {
                        "executable": path_to_exe
                    },
                }
            }
        }

        with open("flow_config.yml", "w") as f:
            f.write(yaml.dump(conf))
        self.monkeypatch.setenv("FLOW_SITE_CONFIG", "flow_config.yml")


    @tmpdir()
    def test_create(self):
        # This test can make do with a mock simulator; - just something executable

        conf = {"versions" : {"2014.2" : {"scalar": {"executable" : "bin/scalar_exe"},
                                          "mpi" : {"executable" : "bin/mpi_exe",
                                                   "mpirun" : "bin/mpirun"}}}}
        with open("ecl100_config.yml","w") as f:
            f.write(yaml.dump(conf))

        os.mkdir("bin")
        self.monkeypatch.setenv("ECL100_SITE_CONFIG", "ecl100_config.yml")
        for f in ["scalar_exe", "mpi_exe", "mpirun"]:
            fname = os.path.join("bin", f)
            with open( fname, "w") as fh:
                fh.write("This is an exectable ...")

            os.chmod(fname, stat.S_IEXEC)

        with open("ECLIPSE.DATA" , "w") as f:
            f.write("Mock eclipse data file")

        ecl_config = Ecl100Config()
        sim = ecl_config.sim("2014.2")
        mpi_sim = ecl_config.mpi_sim("2014.2")
        ecl_run = EclRun("ECLIPSE.DATA", sim)
        self.assertEqual( ecl_run.runPath() , os.getcwd())

        os.mkdir("path")
        with open("path/ECLIPSE.DATA" , "w") as f:
            f.write("Mock eclipse data file")

        ecl_run = EclRun("path/ECLIPSE.DATA", sim)
        self.assertEqual(ecl_run.runPath() , os.path.join(os.getcwd() , "path"))
        self.assertEqual(ecl_run.baseName() , "ECLIPSE")
        self.assertEqual(1, ecl_run.numCpu())

        # invalid number of CPU
        with self.assertRaises(ValueError):
            ecl_run = EclRun("path/ECLIPSE.DATA", sim, num_cpu = "xxx")

        # invalid number of CPU
        with self.assertRaises(Exception):
            ecl_run = EclRun("path/ECLIPSE.DATA", sim, num_cpu = 10)

        ecl_run = EclRun("path/ECLIPSE.DATA", mpi_sim, num_cpu = "10")
        self.assertEqual( 10 , ecl_run.numCpu())

        #Missing datafile
        with self.assertRaises(IOError):
            ecl_run = EclRun("DOES/NOT/EXIST", mpi_sim, num_cpu = "10")

    @pytest.mark.xfail(reason="Finding a version on Komodo of flow that is not OPM-flow")
    @flow_installed
    @tmpdir()
    def test_flow(self):
        self.init_flow_config()
        shutil.copy(os.path.join(self.SOURCE_ROOT , "test-data/local/eclipse/SPE1.DATA"), "SPE1.DATA")
        shutil.copy(os.path.join(self.SOURCE_ROOT , "test-data/local/eclipse/SPE1_ERROR.DATA"), "SPE1_ERROR.DATA")
        flow_config = FlowConfig()
        sim = flow_config.sim()
        flow_run = EclRun("SPE1.DATA", sim)
        flow_run.runEclipse( )

        run(flow_config, ["SPE1.DATA"])

        flow_run = EclRun("SPE1_ERROR.DATA", sim)
        with self.assertRaises(Exception):
           flow_run.runEclipse( )

        run(flow_config, ["SPE1_ERROR.DATA", "--ignore-errors"])

        # Invalid version
        with self.assertRaises(Exception):
            run(flow_config, ["SPE1.DATA", "--version=no/such/version"])

    @tmpdir()
    def test_running_flow_given_env_config_can_still_read_parent_env(self):
        version = "1111.11"

        # create a script that prints env vars ENV1 and ENV2 to a file
        with open("flow", "w") as f:
            f.write("#!/bin/bash\n")
            f.write("echo $ENV1 > out.txt\n")
            f.write("echo $ENV2 >> out.txt\n")
        executable = os.path.join(os.getcwd(), "flow")
        os.chmod(executable, 0o777)

        # create a flow_config.yml with environment extension ENV2
        conf = {
            "default_version": version,
            "versions": {
                version: {
                    "scalar": {
                        "executable": executable,
                        "env": {"ENV2": "VAL2"}
                    },
                }
            }
        }

        with open("flow_config.yml", "w") as f:
            f.write(yaml.dump(conf))

        # set the environment variable ENV1
        self.monkeypatch.setenv("ENV1", "VAL1")
        self.monkeypatch.setenv("FLOW_SITE_CONFIG", "flow_config.yml")

        with open("DUMMY.DATA", "w") as f:
            f.write("dummy")

        with open("DUMMY.PRT", "w") as f:
            f.write("Errors 0\n")
            f.write("Bugs 0\n")

        # run the script
        flow_config = FlowConfig()
        sim = flow_config.sim()
        flow_run = EclRun("DUMMY.DATA", sim)
        flow_run.runEclipse()

        # assert that the script was able to read both the variables correctly
        with open("out.txt") as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].strip(), "VAL1")
        self.assertEqual(lines[1].strip(), "VAL2")

    @tmpdir()
    def test_running_flow_given_no_env_config_can_still_read_parent_env(self):
        version = "1111.11"

        # create a script that prints env vars ENV1 and ENV2 to a file
        with open("flow", "w") as f:
            f.write("#!/bin/bash\n")
            f.write("echo $ENV1 > out.txt\n")
            f.write("echo $ENV2 >> out.txt\n")
        executable = os.path.join(os.getcwd(), "flow")
        os.chmod(executable, 0o777)

        # create a flow_config.yml with environment extension ENV2
        conf = {
            "default_version": version,
            "versions": {
                version: {
                    "scalar": {
                        "executable": executable
                    },
                }
            }
        }

        with open("flow_config.yml", "w") as f:
            f.write(yaml.dump(conf))

        # set the environment variable ENV1
        self.monkeypatch.setenv("ENV1", "VAL1")
        self.monkeypatch.setenv("ENV2", "VAL2")
        self.monkeypatch.setenv("FLOW_SITE_CONFIG", "flow_config.yml")

        with open("DUMMY.DATA", "w") as f:
            f.write("dummy")

        with open("DUMMY.PRT", "w") as f:
            f.write("Errors 0\n")
            f.write("Bugs 0\n")

        # run the script
        flow_config = FlowConfig()
        sim = flow_config.sim()
        flow_run = EclRun("DUMMY.DATA", sim)
        flow_run.runEclipse()

        # assert that the script was able to read both the variables correctly
        with open("out.txt") as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].strip(), "VAL1")
        self.assertEqual(lines[1].strip(), "VAL2")

    @tmpdir()
    def test_running_flow_given_env_variables_with_same_name_as_parent_env_variables_will_overwrite(self):
        version = "1111.11"

        # create a script that prints env vars ENV1 and ENV2 to a file
        with open("flow", "w") as f:
            f.write("#!/bin/bash\n")
            f.write("echo $ENV1 > out.txt\n")
            f.write("echo $ENV2 >> out.txt\n")
        executable = os.path.join(os.getcwd(), "flow")
        os.chmod(executable, 0o777)

        # create a flow_config.yml with environment extension ENV2
        conf = {
            "default_version": version,
            "versions": {
                version: {
                    "scalar": {
                        "executable": executable,
                        "env": {"ENV1": "OVERWRITTEN1", "ENV2": "OVERWRITTEN2"}
                    },
                }
            }
        }

        with open("flow_config.yml", "w") as f:
            f.write(yaml.dump(conf))

        # set the environment variable ENV1
        self.monkeypatch.setenv("ENV1", "VAL1")
        self.monkeypatch.setenv("ENV2", "VAL2")
        self.monkeypatch.setenv("FLOW_SITE_CONFIG", "flow_config.yml")

        with open("DUMMY.DATA", "w") as f:
            f.write("dummy")

        with open("DUMMY.PRT", "w") as f:
            f.write("Errors 0\n")
            f.write("Bugs 0\n")

        # run the script
        flow_config = FlowConfig()
        sim = flow_config.sim()
        flow_run = EclRun("DUMMY.DATA", sim)
        flow_run.runEclipse()

        # assert that the script was able to read both the variables correctly
        with open("out.txt") as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].strip(), "OVERWRITTEN1")
        self.assertEqual(lines[1].strip(), "OVERWRITTEN2")

    @tmpdir()
    @pytest.mark.equinor_test
    def test_run(self):
        self.init_ecl100_config()
        shutil.copy(os.path.join(self.SOURCE_ROOT, "test-data/local/eclipse/SPE1.DATA"), "SPE1.DATA")
        ecl_config = Ecl100Config( )
        sim = ecl_config.sim("2014.2")
        ecl_run = EclRun("SPE1.DATA", sim)
        ecl_run.runEclipse( )

        ok_path = os.path.join(ecl_run.runPath(), "{}.OK".format(ecl_run.baseName()))
        log_path = os.path.join(ecl_run.runPath(), "{}.LOG".format(ecl_run.baseName()))

        self.assertTrue(os.path.isfile(ok_path))
        self.assertTrue(os.path.isfile(log_path))
        self.assertTrue(os.path.getsize(log_path) > 0)

        errors = ecl_run.parseErrors( )
        self.assertEqual( 0 , len(errors ))

        # Monkey patching the ecl_run to use an executable which
        # will fail with exit(1); don't think Eclipse actually
        # fails with exit(1) - but let us at least be prepared
        # when/if it does.
        ecl_run.sim.executable = os.path.join( self.SOURCE_ROOT , "python/tests/res/fm/ecl_run_fail")
        with self.assertRaises(Exception):
            ecl_run.runEclipse( )

    @pytest.mark.equinor_test
    @tmpdir()
    def test_run_api(self):
        self.init_ecl100_config()
        shutil.copy(os.path.join(self.SOURCE_ROOT, "test-data/local/eclipse/SPE1.DATA"), "SPE1.DATA")
        ecl_config = Ecl100Config( )
        run(ecl_config, ["SPE1.DATA", "--version=2014.2"])

        self.assertTrue(os.path.isfile("SPE1.DATA"))


    @pytest.mark.equinor_test
    @tmpdir()
    def test_failed_run(self):
        self.init_ecl100_config()
        shutil.copy(os.path.join(self.SOURCE_ROOT, "test-data/local/eclipse/SPE1_ERROR.DATA"), "SPE1_ERROR.DATA")
        ecl_config = Ecl100Config()
        sim = ecl_config.sim("2014.2")
        ecl_run = EclRun("SPE1_ERROR", sim)
        with self.assertRaises(Exception):
            ecl_run.runEclipse( )
        try:
            ecl_run.runEclipse( )
        except Exception as e:
            self.assertTrue( "ERROR" in str(e) )


    @pytest.mark.equinor_test
    @tmpdir()
    def test_failed_run_OK(self):
        self.init_ecl100_config()
        shutil.copy(os.path.join(self.SOURCE_ROOT, "test-data/local/eclipse/SPE1_ERROR.DATA"), "SPE1_ERROR.DATA")
        ecl_config = Ecl100Config()
        run(ecl_config, ["SPE1_ERROR", "--version=2014.2", "--ignore-errors"])

        # Monkey patching the ecl_run to use an executable which will fail with exit(1),
        # in the nocheck mode that should also be OK.
        sim = ecl_config.sim("2014.2")
        ecl_run = EclRun("SPE1_ERROR", sim, check_status = False)
        ecl_run.sim.executable = os.path.join(self.SOURCE_ROOT, "python/tests/res/fm/ecl_run_fail")
        ecl_run.runEclipse()


    @pytest.mark.equinor_test
    @tmpdir()
    def test_mpi_run(self):
        self.init_ecl100_config()
        shutil.copy(os.path.join(self.SOURCE_ROOT, "test-data/local/eclipse/SPE1_PARALLELL.DATA"), "SPE1_PARALLELL.DATA")
        ecl_config = Ecl100Config()
        run(ecl_config, ["SPE1_PARALLELL.DATA", "--version=2014.2", "--num-cpu=2"])
        self.assertTrue(os.path.isfile("SPE1_PARALLELL.LOG"))
        self.assertTrue(os.path.getsize("SPE1_PARALLELL.LOG") > 0)


    @pytest.mark.equinor_test
    @tmpdir()
    def test_summary_block(self):
        self.init_ecl100_config()
        shutil.copy(os.path.join(self.SOURCE_ROOT, "test-data/local/eclipse/SPE1.DATA"), "SPE1.DATA")
        ecl_config = Ecl100Config()
        sim = ecl_config.sim("2014.2")
        ecl_run = EclRun("SPE1.DATA", sim)
        ret_value = ecl_run.summary_block( )
        self.assertTrue( ret_value is None )

        ecl_run.runEclipse( )
        ecl_sum = ecl_run.summary_block( )
        self.assertTrue(isinstance(ecl_sum, EclSum))


    @pytest.mark.equinor_test
    @tmpdir()
    def test_check(self):
        full_case   = os.path.join(self.SOURCE_ROOT, "test-data/Equinor/ECLIPSE/Gurbat/ECLIPSE" )
        short_case  = os.path.join(self.SOURCE_ROOT, "test-data/Equinor/ECLIPSE/ShortSummary/ECLIPSE" )
        failed_case = os.path.join(self.SOURCE_ROOT, "test-data/Equinor/ECLIPSE/SummaryFail/NOR-2013A_R002_1208-0")

        with self.assertRaises(IOError):
            self.assertTrue( EclRun.checkCase( full_case , failed_case ))

        with self.assertRaises(IOError):
            self.assertTrue( EclRun.checkCase( full_case , "DOES-NOT-EXIST" ))

        with self.assertRaises(IOError):
            self.assertTrue( EclRun.checkCase( "DOES-NOT-EXIST" , full_case))

        with self.assertRaises(ValueError):
            EclRun.checkCase(full_case, short_case)

        self.assertTrue(not os.path.isfile("CHECK_ECLIPSE_RUN.OK"))
        self.assertTrue( EclRun.checkCase( full_case , full_case ))
        self.assertTrue( os.path.isfile("CHECK_ECLIPSE_RUN.OK"))

        os.remove("CHECK_ECLIPSE_RUN.OK")
        self.assertTrue( EclRun.checkCase( short_case , full_case ))   # Simulation is longer than refcase - OK
        self.assertTrue( os.path.isfile("CHECK_ECLIPSE_RUN.OK"))


    @pytest.mark.equinor_test
    @tmpdir()
    def test_error_parse(self):
        self.init_ecl100_config()
        shutil.copy(os.path.join(self.SOURCE_ROOT, "test-data/local/eclipse/SPE1.DATA"), "SPE1.DATA")
        prt_file = os.path.join(self.SOURCE_ROOT, "test-data/local/eclipse/parse/ERROR.PRT")
        shutil.copy(prt_file, "SPE1.PRT")

        ecl_config = Ecl100Config()
        sim = ecl_config.sim("2014.2")
        ecl_run = EclRun("SPE1.DATA", sim)

        error_list = ecl_run.parseErrors()
        self.assertEqual( len(error_list), 2)


        # NB: The ugly white space in the error0 literal is actually part of
        #     the string we are matching; i.e. it must be retained.
        error0 = """ @--  ERROR  AT TIME        0.0   DAYS    ( 1-JAN-0):
 @           UNABLE TO OPEN INCLUDED FILE                                    
 @           /private/joaho/ERT/git/Gurbat/XXexample_grid_sim.GRDECL         
 @           SYSTEM ERROR CODE IS       29                                   """

        error1 = """ @--  ERROR  AT TIME        0.0   DAYS    ( 1-JAN-0):
 @           INCLUDE FILES MISSING.                                          """

        self.assertEqual( error_list[0], error0)
        self.assertEqual( error_list[1], error1)
