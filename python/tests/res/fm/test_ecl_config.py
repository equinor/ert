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
import inspect
import unittest
import yaml
from ecl.util.test import TestAreaContext
from tests import ResTest
from _pytest.monkeypatch import MonkeyPatch

from res.fm.ecl import Ecl100Config
from res.fm.ecl.ecl_config import Keys


class EclConfigTest(ResTest):
    def setUp(self):
        self.ecl_config_path = os.path.dirname(inspect.getsourcefile(Ecl100Config))
        self.monkeypatch = MonkeyPatch()

    def tearDown(self):
        self.monkeypatch.undo()

    def test_load(self):
        self.monkeypatch.setenv("ECL100_SITE_CONFIG", "file/does/not/exist")
        with self.assertRaises(IOError):
            conf = Ecl100Config()

        self.monkeypatch.setenv(
            "ECL100_SITE_CONFIG",
            os.path.join(self.ecl_config_path, "ecl100_config.yml"),
        )
        conf = Ecl100Config()

        with TestAreaContext("yaml_invalid"):
            with open("file.yml", "w") as f:
                f.write("this:\n -should\n-be\ninvalid:yaml?")

            self.monkeypatch.setenv("ECL100_SITE_CONFIG", "file.yml")
            with self.assertRaises(ValueError):
                conf = Ecl100Config()

            scalar_path = "scalar"
            scalar_exe = "bin/scalar_exe"
            mpi_exe = "bin/mpi_exe"
            mpi_run = "bin/mpi_run"

            os.mkdir("bin")
            for f in ["scalar_exe", "mpi_exe", "mpi_run"]:
                fname = os.path.join("bin", f)
                with open(fname, "w") as fh:
                    fh.write("This is an exectable ...")

                os.chmod(fname, stat.S_IEXEC)

            intel_path = "intel"
            self.monkeypatch.setenv("ENV1", "A")
            self.monkeypatch.setenv("ENV2", "C")
            d = {
                Keys.env: {"LICENSE_SERVER": "license@company.com"},
                Keys.versions: {
                    "2015": {
                        Keys.scalar: {Keys.executable: scalar_exe},
                        Keys.mpi: {
                            Keys.executable: mpi_exe,
                            Keys.mpirun: mpi_run,
                            Keys.env: {
                                "I_MPI_ROOT": "$ENV1:B:$ENV2",
                                "TEST_VAR": "$ENV1.B.$ENV2 $UNKNOWN_VAR",
                                "P4_RSHCOMMAND": "",
                                "LD_LIBRARY_PATH": "{}:$LD_LIBRARY_PATH".format(
                                    intel_path
                                ),
                                "PATH": "{}/bin64:$PATH".format(intel_path),
                            },
                        },
                    },
                    "2016": {
                        Keys.scalar: {Keys.executable: "/does/not/exist"},
                        Keys.mpi: {
                            Keys.executable: "/does/not/exist",
                            Keys.mpirun: mpi_run,
                        },
                    },
                    "2017": {
                        Keys.mpi: {
                            Keys.executable: mpi_exe,
                            Keys.mpirun: "/does/not/exist",
                        }
                    },
                },
            }

            with open("file.yml", "w") as f:
                f.write(yaml.dump(d))

            conf = Ecl100Config()
            # Fails because there is no version 2020
            with self.assertRaises(KeyError):
                sim = conf.sim("2020")

            # Fails because the 2016 version points to a not existing executable
            with self.assertRaises(OSError):
                sim = conf.sim("2016")

            # Fails because the 2016 mpi version points to a non existing mpi executable
            with self.assertRaises(OSError):
                sim = conf.mpi_sim("2016")

            # Fails because the 2017 mpi version mpirun points to a non existing mpi executable
            with self.assertRaises(OSError):
                sim = conf.mpi_sim("2017")

            # Fails because the 2017 scalar version is not registered
            with self.assertRaises(KeyError):
                sim = conf.sim("2017")

            sim = conf.sim("2015")
            mpi_sim = conf.mpi_sim("2015")

            # Check that global environment has been propagated down.
            self.assertIn("LICENSE_SERVER", mpi_sim.env)

            # Check replacement of $ENV_VAR in values.
            self.assertEqual(mpi_sim.env["I_MPI_ROOT"], "A:B:C")
            self.assertEqual(mpi_sim.env["TEST_VAR"], "A.B.C $UNKNOWN_VAR")
            self.assertEqual(len(mpi_sim.env), 1 + 5)

            sim = conf.sim("2015")
            self.assertEqual(sim.executable, scalar_exe)
            self.assertIsNone(sim.mpirun)

            with self.assertRaises(Exception):
                simulators = conf.simulators()

            simulators = conf.simulators(strict=False)
            self.assertEqual(len(simulators), 2)

    def test_default(self):
        with TestAreaContext("default"):
            os.mkdir("bin")
            scalar_exe = "bin/scalar_exe"
            with open(scalar_exe, "w") as fh:
                fh.write("This is an exectable ...")
            os.chmod(scalar_exe, stat.S_IEXEC)

            d0 = {
                Keys.versions: {
                    "2015": {Keys.scalar: {Keys.executable: scalar_exe}},
                    "2016": {Keys.scalar: {Keys.executable: scalar_exe}},
                }
            }

            d1 = {
                Keys.default_version: "2015",
                Keys.versions: {
                    "2015": {Keys.scalar: {Keys.executable: scalar_exe}},
                    "2016": {Keys.scalar: {Keys.executable: scalar_exe}},
                },
            }

            self.monkeypatch.setenv("ECL100_SITE_CONFIG", os.path.join("file.yml"))
            with open("file.yml", "w") as f:
                f.write(yaml.dump(d1))

            conf = Ecl100Config()
            sim = conf.sim()
            self.assertEqual(sim.version, "2015")
            self.assertIn("2015", conf)
            self.assertNotIn("xxxx", conf)
            self.assertIn(Keys.default, conf)
            self.assertIn(None, conf)

            sim = conf.sim("default")
            self.assertEqual(sim.version, "2015")

            with open("file.yml", "w") as f:
                f.write(yaml.dump(d0))

            conf = Ecl100Config()
            self.assertNotIn(Keys.default, conf)
            self.assertIsNone(conf.default_version)

            with self.assertRaises(Exception):
                sim = conf.sim()

            with self.assertRaises(Exception):
                sim = conf.sim(Keys.default)


if __name__ == "__main__":
    unittest.main()
