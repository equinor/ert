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
import inspect
import unittest
import yaml
from ecl.util.test import TestAreaContext
from tests import ResTest

from res.fm.ecl import EclConfig


class EclConfigTest(ResTest):

    def setUp(self):
        self.ecl_config_path = os.path.dirname( inspect.getsourcefile(EclConfig) )

    def test_load(self):
        os.environ["ECL_SITE_CONFIG"] = "file/does/not/exist"
        with self.assertRaises(IOError):
            conf = EclConfig()

        os.environ["ECL_SITE_CONFIG"] = os.path.join(self.ecl_config_path, "ecl_config.yml")
        conf = EclConfig()

        with TestAreaContext("yaml_invalid"):
            with open("file.yml","w") as f:
                f.write("this:\n -should\n-be\ninvalid:yaml?")

            os.environ["ECL_SITE_CONFIG"] = "file.yml"
            with self.assertRaises(ValueError):
                conf = EclConfig()

            scalar_path = "scalar"
            scalar_exe = "bin/scalar_exe"
            mpi_exe = "bin/mpi_exe"
            mpi_run = "bin/mpi_run"

            os.mkdir("bin")
            for f in ["scalar_exe", "mpi_exe", "mpi_run"]:
                fname = os.path.join("bin", f)
                with open( fname, "w") as fh:
                    fh.write("This is an exectable ...")

                os.chmod(fname, stat.S_IEXEC)

            intel_path = "intel"
            os.environ["ENV1"] = "A"
            os.environ["ENV2"] = "C"
            d = {"env" : {"LICENSE_SERVER" : "license@company.com"},
                 "simulators": {"ecl100" : {"2015" : {"scalar": {"executable" : scalar_exe},
                                                      "mpi"   : {"executable" : mpi_exe,
                                                                 "mpirun"     : mpi_run,
                                                                 "env" : {"I_MPI_ROOT" : "$ENV1:B:$ENV2",
                                                                          "TEST_VAR" : "$ENV1.B.$ENV2 $UNKNOWN_VAR",
                                                                          "P4_RSHCOMMAND" : "",
                                                                          "LD_LIBRARY_PATH" : "{}:$LD_LIBRARY_PATH".format(intel_path),
                                                                          "PATH" : "{}/bin64:$PATH".format(intel_path)}}}}
                                ,
                                "flow" : {"2018.04" : {"scalar" : {"executable" : "/does/not/exist"},
                                                       "mpi" : {"executable" : mpi_exe,
                                                                "mpirun"  : "/does/not/exist"}}}}}

            with open("file.yml", "w") as f:
                f.write( yaml.dump(d) )

            conf = EclConfig()
            # Fails because there is no simulator ecl99
            with self.assertRaises(KeyError):
                sim = conf.sim("ecl99", "2015")

            # Fails because there is no version 2020 for ecl100
            with self.assertRaises(KeyError):
                sim = conf.sim("ecl100", "2020")

            # Fails because the 2018.04 version of flow points to a not existing executable
            with self.assertRaises(OSError):
                sim = conf.sim("flow", "2018.04")

            # Fails because the 2018.04 mpi version points to a non existing mpirun binary
            with self.assertRaises(OSError):
                sim = conf.mpi_sim("flow", "2018.04")

            with self.assertRaises(Exception):
                conf.sim("flowIx", "2018.04")

            with self.assertRaises(Exception):
                conf.mpi_sim("flow", "2018.04")

            mpi_sim = conf.mpi_sim("ecl100", "2015")
            # Check that global environment has been propagated down.
            self.assertIn("LICENSE_SERVER", mpi_sim.env)

            # Check replacement of $ENV_VAR in values.
            self.assertEqual(mpi_sim.env["I_MPI_ROOT"], "A:B:C")
            self.assertEqual(mpi_sim.env["TEST_VAR"], "A.B.C $UNKNOWN_VAR")
            self.assertEqual(len(mpi_sim.env), 1 + 5)

            sim = conf.sim("ecl100", "2015")
            self.assertEqual(sim.executable, scalar_exe)
            self.assertIsNone(sim.mpirun)

            with self.assertRaises(Exception):
                simulators = conf.simulators()

            simulators = conf.simulators(strict = False)
            self.assertEqual(len(simulators), 2)



if __name__ == "__main__":
    unittest.main()
