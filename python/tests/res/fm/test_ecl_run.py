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

from ecl.summary import EclSum
from ecl.util.test import TestAreaContext
from tests import ResTest, equinor_test
from res.fm.ecl import *



flow_config = FlowConfig( )
try:
    sim = flow_config.sim()
    have_flow = True
except:
    have_flow = False

class EclRunTest(ResTest):
    def setUp(self):
        self.ecl_config_path = os.path.dirname( inspect.getsourcefile(Ecl100Config) )


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
        os.environ["ECL100_SITE_CONFIG"] = "ecl100_config.yml"


    def test_create(self):
        # This test can make do with a mock simulator; - just something executable

        with TestAreaContext("ecl_run"):
            conf = {"versions" : {"2014.2" : {"scalar": {"executable" : "bin/scalar_exe"},
                                              "mpi" : {"executable" : "bin/mpi_exe",
                                                       "mpirun" : "bin/mpirun"}}}}
            with open("ecl100_config.yml","w") as f:
                f.write( yaml.dump(conf) )

            os.mkdir("bin")
            os.environ["ECL100_SITE_CONFIG"] = "ecl100_config.yml"
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
            self.assertEqual( ecl_run.runPath() , os.path.join(os.getcwd() , "path"))
            self.assertEqual( ecl_run.baseName() , "ECLIPSE")
            self.assertEqual( 1 , ecl_run.numCpu())

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



    @unittest.skipUnless(have_flow, "Requires flow")
    def test_flow(self):
        with TestAreaContext("ecl_run") as ta:
            ta.copy_file( os.path.join(self.SOURCE_ROOT , "test-data/local/eclipse/SPE1.DATA"))
            ta.copy_file( os.path.join(self.SOURCE_ROOT , "test-data/local/eclipse/SPE1_ERROR.DATA"))
            os.makedirs("ecl_run")
            shutil.move("SPE1.DATA", "ecl_run")
            shutil.move("SPE1_ERROR.DATA" , "ecl_run")
            flow_config = FlowConfig()
            sim = flow_config.sim()
            flow_run = EclRun("ecl_run/SPE1.DATA", sim)
            flow_run.runEclipse( )

            run(flow_config, ["ecl_run/SPE1.DATA"])

            flow_run = EclRun("ecl_run/SPE1_ERROR.DATA", sim)
            with self.assertRaises(Exception):
               flow_run.runEclipse( )

            run(flow_config, ["ecl_run/SPE1_ERROR.DATA", "--ignore-errors"])

            # Invalid version
            with self.assertRaises(Exception):
                run(flow_config, ["ecl_run/SPE1.DATA", "--version=no/such/version"])



    @equinor_test()
    def test_run(self):
        with TestAreaContext("ecl_run") as ta:
            self.init_ecl100_config()
            ta.copy_file( os.path.join(self.SOURCE_ROOT , "test-data/local/eclipse/SPE1.DATA"))
            os.makedirs("ecl_run")
            shutil.move("SPE1.DATA" , "ecl_run")
            ecl_config = Ecl100Config( )
            sim = ecl_config.sim("2014.2")
            ecl_run = EclRun("ecl_run/SPE1.DATA", sim)
            ecl_run.runEclipse( )

            self.assertTrue( os.path.isfile( os.path.join( ecl_run.runPath() , "%s.stderr" % ecl_run.baseName())))
            self.assertTrue( os.path.isfile( os.path.join( ecl_run.runPath() , "%s.LOG" % ecl_run.baseName())))
            self.assertTrue( os.path.isfile( os.path.join( ecl_run.runPath() , "%s.OK" % ecl_run.baseName())))

            errors = ecl_run.parseErrors( )
            self.assertEqual( 0 , len(errors ))

            # Monkey patching the ecl_run to use an executable which
            # will fail with exit(1); don't think Eclipse actually
            # fails with exit(1) - but let us at least be prepared
            # when/if it does.
            ecl_run.sim.executable = os.path.join( self.SOURCE_ROOT , "python/tests/res/fm/ecl_run_fail")
            with self.assertRaises(Exception):
                ecl_run.runEclipse( )

    @equinor_test()
    def test_run_api(self):
        with TestAreaContext("ecl_run_api") as ta:
            self.init_ecl100_config()
            ta.copy_file( os.path.join(self.SOURCE_ROOT , "test-data/local/eclipse/SPE1.DATA"))
            os.makedirs("ecl_run")
            shutil.move("SPE1.DATA" , "ecl_run")
            ecl_config = Ecl100Config( )
            run(ecl_config, ["ecl_run/SPE1.DATA", "--version=2014.2"])

            self.assertTrue( os.path.isfile("ecl_run/SPE1.DATA"))
            self.assertTrue( os.path.isfile("ecl_run/SPE1.DATA"))
            self.assertTrue( os.path.isfile("ecl_run/SPE1.DATA"))




    @equinor_test()
    def test_failed_run(self):
        with TestAreaContext("ecl_run") as ta:
            self.init_ecl100_config()
            ta.copy_file( os.path.join(self.SOURCE_ROOT , "test-data/local/eclipse/SPE1_ERROR.DATA"))
            ecl_config = Ecl100Config()
            sim = ecl_config.sim("2014.2")
            ecl_run = EclRun("SPE1_ERROR", sim)
            with self.assertRaises(Exception):
                ecl_run.runEclipse( )

            try:
                ecl_run.runEclipse( )
            except Exception as e:
                self.assertTrue( "ERROR" in str(e) )


    @equinor_test()
    def test_failed_run_OK(self):
        with TestAreaContext("ecl_run") as ta:
            self.init_ecl100_config()
            ta.copy_file( os.path.join(self.SOURCE_ROOT , "test-data/local/eclipse/SPE1_ERROR.DATA"))
            ecl_config = Ecl100Config()
            run(ecl_config, ["SPE1_ERROR", "--version=2014.2", "--ignore-errors"])

            # Monkey patching the ecl_run to use an executable which will fail with exit(1),
            # in the nocheck mode that should also be OK.
            sim = ecl_config.sim("2014.2")
            ecl_run = EclRun("SPE1_ERROR", sim, check_status = False)
            ecl_run.sim.executable = os.path.join( self.SOURCE_ROOT , "python/tests/res/fm/ecl_run_fail")
            ecl_run.runEclipse( )


    @equinor_test()
    def test_mpi_run(self):
        with TestAreaContext("ecl_run") as ta:
            self.init_ecl100_config()
            ta.copy_file( os.path.join(self.SOURCE_ROOT , "test-data/local/eclipse/SPE1_PARALLELL.DATA"))
            ecl_config = Ecl100Config()
            run(ecl_config, ["SPE1_PARALLELL.DATA", "--version=2014.2", "--num-cpu=2"])
            self.assertTrue( os.path.isfile( "SPE1_PARALLELL.stderr"))
            self.assertTrue( os.path.isfile( "SPE1_PARALLELL.LOG"))


    @equinor_test()
    def test_summary_block(self):
        with TestAreaContext("ecl_run") as ta:
            self.init_ecl100_config()
            ta.copy_file( os.path.join(self.SOURCE_ROOT , "test-data/local/eclipse/SPE1.DATA"))
            ecl_config = Ecl100Config()
            sim = ecl_config.sim("2014.2")
            ecl_run = EclRun("SPE1.DATA", sim)
            ret_value = ecl_run.summary_block( )
            self.assertTrue( ret_value is None )

            ecl_run.runEclipse( )
            ecl_sum = ecl_run.summary_block( )
            self.assertTrue(isinstance(ecl_sum, EclSum))


    @equinor_test()
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
            EclRun.checkCase( full_case , short_case )

        with TestAreaContext("ecl_check1"):
            self.assertTrue( EclRun.checkCase( full_case , full_case ))
            self.assertTrue( os.path.isfile("CHECK_ECLIPSE_RUN.OK"))

        with TestAreaContext("ecl_check2"):
            self.assertTrue( EclRun.checkCase( short_case , full_case ))   # Simulation is longer than refcase - OK
            self.assertTrue( os.path.isfile("CHECK_ECLIPSE_RUN.OK"))



    @equinor_test()
    def test_error_parse(self):
        with TestAreaContext("ecl_run") as ta:
            self.init_ecl100_config()
            ta.copy_file( os.path.join(self.SOURCE_ROOT , "test-data/local/eclipse/SPE1.DATA"))
            prt_file = os.path.join(self.SOURCE_ROOT , "test-data/local/eclipse/parse/ERROR.PRT")
            shutil.copy(prt_file , "SPE1.PRT")

            ecl_config = Ecl100Config()
            sim = ecl_config.sim("2014.2")
            ecl_run = EclRun("SPE1.DATA", sim)

            error_list = ecl_run.parseErrors( )
            self.assertEqual( len(error_list) , 2 )


            # NB: The ugly white space in the error0 literal is actually part of
            #     the string we are matching; i.e. it must be retained.
            error0 = """ @--  ERROR  AT TIME        0.0   DAYS    ( 1-JAN-0):
 @           UNABLE TO OPEN INCLUDED FILE                                    
 @           /private/joaho/ERT/git/Gurbat/XXexample_grid_sim.GRDECL         
 @           SYSTEM ERROR CODE IS       29                                   """

            error1 = """ @--  ERROR  AT TIME        0.0   DAYS    ( 1-JAN-0):
 @           INCLUDE FILES MISSING.                                          """
            
            self.assertEqual( error_list[0] , error0 )
            self.assertEqual( error_list[1] , error1 )

