#  Copyright (C) 2018  Statoil ASA, Norway.
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
import yaml

from ecl.summary import EclSum
from ecl.util.test import TestAreaContext
from tests import ResTest, statoil_test
from res.fm.ecl import EclConfig, EclRun



class EclRunTest(ResTest):
    def setUp(self):
        os.environ["ECL_SITE_CONFIG"] = os.path.join(self.SOURCE_ROOT, "python/res/fm/ecl/ecl_config.yml")


    def init_config(self):
        conf = {"env" : {"F_UFMTENDIAN" : "big",
                         "LM_LICENSE_FILE" : "7321@eclipse-lic-no.statoil.no",
                         "ARCH" : "x86_64"},
                "simulators" : {"flow"   : {"daily"  : {"scalar": {"executable" : "/project/res/x86_64_RH_6/bin/flowdaily"}}},
                                "ecl100" : {"2014.2" : {"scalar": {"executable" : "/prog/ecl/grid/2014.2/bin/linux_x86_64/eclipse.exe"},
                                                        "mpi"   : {"executable" : "/prog/ecl/grid/2014.2/bin/linux_x86_64/eclipse_ilmpi.exe",
                                                                   "mpirun" : "/prog/ecl/grid/tools/linux_x86_64/intel/mpi/5.0.2.044/bin64/mpirun",
                                                                   "env" : {"I_MPI_ROOT" : "/prog/ecl/grid/tools/linux_x86_64/intel/mpi/5.0.2.044/",
                                                                            "P4_RSHCOMMAND" : "ssh",
                                                                            "LD_LIBRARY_PATH" : "/prog/ecl/grid/tools/linux_x86_64/intel/mpi/5.0.2.044/lib64:$LD_LIBRARY_PATH",
                                                                            "PATH" : "/prog/ecl/grid/tools/linux_x86_64/intel/mpi/5.0.2.044/bin64:$PATH"}}}}}}
        with open("ecl_config.yml","w") as f:
            f.write( yaml.dump(conf) )
        os.environ["ECL_SITE_CONFIG"] = "ecl_config.yml"


    def test_create(self):
        # This test can make do with a mock simulator; - just something executable

        with TestAreaContext("ecl_run"):
            self.init_config()
            conf = {"simulators" : {"ecl100" : {"2014.2" : {"scalar": {"executable" : "bin/scalar_exe"},
                                                            "mpi" : {"executable" : "bin/mpi_exe",
                                                                     "mpirun" : "bin/mpirun"}}}}}
            with open("ecl_config.yml","w") as f:
                f.write( yaml.dump(conf) )

            os.mkdir("bin")
            EclConfig.config_file = "ecl_config.yml"
            for f in ["scalar_exe", "mpi_exe", "mpirun"]:
                fname = os.path.join("bin", f)
                with open( fname, "w") as fh:
                    fh.write("This is an exectable ...")

                os.chmod(fname, stat.S_IEXEC)

            # Wrong arg count
            with self.assertRaises(ValueError):
                argv = []
                ecl_run = EclRun(argv)

            # Wrong arg count
            with self.assertRaises(ValueError):
                argv = [1,2,3,4,5]
                ecl_run = EclRun(argv)


            with open("ECLIPSE.DATA" , "w") as f:
                f.write("Mock eclipse data file")

            #Unknown simulator in argv[0]
            with self.assertRaises(ValueError):
                argv = ["Simulator" , "2014.2" , "ECLIPSE.DATA"]
                ecl_run = EclRun(argv)

            ecl_run = EclRun(["run_ecl100" , "2014.2" , "ECLIPSE.DATA"])
            self.assertEqual( ecl_run.runPath() , os.getcwd())

            os.mkdir("path")
            with open("path/ECLIPSE.DATA" , "w") as f:
                f.write("Mock eclipse data file")

            ecl_run = EclRun(["run_ecl100" , "2014.2" , "path/ECLIPSE.DATA"])
            self.assertEqual( ecl_run.runPath() , os.path.join(os.getcwd() , "path"))
            self.assertEqual( ecl_run.baseName() , "ECLIPSE")


            argv = ["run_ecl100" , "2014.2" , "ECLIPSE.DATA"]
            ecl_run = EclRun(argv)
            self.assertEqual( 1 , ecl_run.numCpu())

            # invalid number of CPU
            with self.assertRaises(ValueError):
                argv = ["run_ecl100" , "2014.2" , "ECLIPSE.DATA" , "xx"]
                ecl_run = EclRun(argv)

            argv = ["run_ecl100" , "2014.2" , "ECLIPSE.DATA" , "10"]
            ecl_run = EclRun(argv)
            self.assertEqual( 10 , ecl_run.numCpu())

            argv = ["run_ecl100" , "MISSING_VERSION" , "ECLIPSE.DATA" , "10"]
            with self.assertRaises(KeyError):
                ecl_run = EclRun(argv)

            #Missing datafile
            with self.assertRaises(IOError):
                argv = ["run_ecl100" , "2014.2" , "ECLIPSE_DOES_NOT_EXIST.DATA"]
                ecl_run = EclRun(argv)


    @statoil_test()
    def test_run(self):
        with TestAreaContext("ecl_run") as ta:
            self.init_config()
            ta.copy_file( os.path.join(self.SOURCE_ROOT , "test-data/local/eclipse/SPE1.DATA"))
            os.makedirs("ecl_run")
            shutil.move("SPE1.DATA" , "ecl_run")
            argv = ["run_ecl100" , "2014.2" , "ecl_run/SPE1.DATA"]
            ecl_run = EclRun(argv)
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
            ecl_run.sim.executable = os.path.join( self.SOURCE_ROOT , "tests/classes/ecl_run_fail")
            with self.assertRaises(Exception):
                ecl_run.runEclipse( )


    @statoil_test()
    def test_flow(self):
        with TestAreaContext("ecl_run") as ta:
            self.init_config()
            ta.copy_file( os.path.join(self.SOURCE_ROOT , "test-data/local/eclipse/SPE1.DATA"))
            ta.copy_file( os.path.join(self.SOURCE_ROOT , "test-data/local/eclipse/SPE1.ERROR.DATA"))
            os.makedirs("ecl_run")
            shutil.move("SPE1.DATA" , "ecl_run")
            argv = ["run_flow" , "daily" , "ecl_run/SPE1.DATA"]
            flow_run = EclRun(argv)
            flow_run.runEclipse( )

            flow_run = EclRun(["run_flow" , "daily" , "SPE1.ERROR.DATA"])
            with self.assertRaises(Exception):
                flow_run.runEclipse( )


    @statoil_test()
    def test_failed_run(self):
        with TestAreaContext("ecl_run") as ta:
            self.init_config()
            ta.copy_file( os.path.join(self.SOURCE_ROOT , "test-data/local/eclipse/SPE1.ERROR.DATA"))
            argv = ["run_ecl100" , "2014.2" , "SPE1.ERROR"]
            ecl_run = EclRun(argv)
            with self.assertRaises(Exception):
                ecl_run.runEclipse( )

            try:
                ecl_run.runEclipse( )
            except Exception as e:
                self.assertTrue( "ERROR" in str(e) )


    @statoil_test()
    def test_failed_run_OK(self):
        with TestAreaContext("ecl_run") as ta:
            self.init_config()
            ta.copy_file( os.path.join(self.SOURCE_ROOT , "test-data/local/eclipse/SPE1.ERROR.DATA"))
            argv = ["run_ecl100_nocheck" , "2014.2" , "SPE1.ERROR"]
            ecl_run = EclRun(argv)
            ecl_run.runEclipse( )

            # Monkey patching the ecl_run to use an executable which will fail with exit(1),
            # in the nocheck mode that should also be OK.
            ecl_run.sim.executable = os.path.join( self.SOURCE_ROOT , "python/tests/res/fm/ecl_run_fail")
            ecl_run.runEclipse( )


    @statoil_test()
    def test_mpi_run(self):
        with TestAreaContext("ecl_run") as ta:
            self.init_config()
            ta.copy_file( os.path.join(self.SOURCE_ROOT , "test-data/local/eclipse/SPE1_PARALLELL.DATA"))
            argv = ["run_ecl100" , "2014.2" , "SPE1_PARALLELL.DATA" , "2"]
            ecl_run = EclRun(argv)
            ecl_run.runEclipse( )
            self.assertTrue( os.path.isfile( os.path.join( ecl_run.runPath() , "%s.stderr" % ecl_run.baseName())))
            self.assertTrue( os.path.isfile( os.path.join( ecl_run.runPath() , "%s.LOG" % ecl_run.baseName())))



    @statoil_test()
    def test_summary_block(self):
        with TestAreaContext("ecl_run") as ta:
            self.init_config()
            ta.copy_file( os.path.join(self.SOURCE_ROOT , "test-data/local/eclipse/SPE1.ERROR.DATA"))
            ta.copy_file( os.path.join(self.SOURCE_ROOT , "test-data/local/eclipse/SPE1.DATA"))
            argv = ["run_ecl100" , "2014.2" , "SPE1"]
            ecl_run = EclRun(argv)
            ret_value = ecl_run.summary_block( )
            self.assertTrue( ret_value is None )

            ecl_run.runEclipse( )
            ecl_sum = ecl_run.summary_block( )
            self.assertTrue(isinstance(ecl_sum, EclSum))


    @statoil_test()
    def test_check(self):
        full_case   = os.path.join(self.SOURCE_ROOT, "test-data/Statoil/ECLIPSE/Gurbat/ECLIPSE" )
        short_case  = os.path.join(self.SOURCE_ROOT, "test-data/Statoil/ECLIPSE/ShortSummary/ECLIPSE" )
        failed_case = os.path.join(self.SOURCE_ROOT, "test-data/Statoil/ECLIPSE/SummaryFail/NOR-2013A_R002_1208-0")

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



    @statoil_test()
    def test_error_parse(self):
        with TestAreaContext("ecl_run") as ta:
            self.init_config()
            ta.copy_file( os.path.join(self.SOURCE_ROOT , "test-data/local/eclipse/SPE1.ERROR.DATA"))
            ta.copy_file( os.path.join(self.SOURCE_ROOT , "test-data/local/eclipse/SPE1.DATA"))

            argv = ["run_ecl100" , "2014.2" , "SPE1.DATA"]
            ecl_run = EclRun(argv)
            ecl_run.runEclipse( )

            prt_file = os.path.join(self.SOURCE_ROOT , "test-data/local/eclipse/parse/ERROR.PRT")
            shutil.copy(prt_file , "SPE1.PRT")

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

