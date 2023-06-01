import inspect
import os
import stat

import pytest
import yaml

from tests.utils import SOURCE_DIR

from ._import_from_location import import_from_location

# import ecl_config and ecl_run.py from ert/forward-models/res/script
# package-data path which. These are kept out of the ert package to avoid the
# overhead of importing ert. This is necessary as these may be invoked as a
# subprocess on each realization.


ecl_config = import_from_location(
    "ecl_config",
    os.path.join(
        SOURCE_DIR, "src/ert/shared/share/ert/forward-models/res/script/ecl_config.py"
    ),
)

ecl_run = import_from_location(
    "ecl_run",
    os.path.join(
        SOURCE_DIR, "src/ert/shared/share/ert/forward-models/res/script/ecl_run.py"
    ),
)


@pytest.mark.usefixtures("use_tmpdir")
def test_loading_of_eclipse_configurations(monkeypatch):
    source_file = inspect.getsourcefile(ecl_config.Ecl100Config)
    assert source_file is not None
    ecl_config_path = os.path.dirname(source_file)
    monkeypatch.setenv("ECL100_SITE_CONFIG", "file/does/not/exist")
    with pytest.raises(IOError):
        conf = ecl_config.Ecl100Config()

    monkeypatch.setenv(
        "ECL100_SITE_CONFIG",
        os.path.join(ecl_config_path, "ecl100_config.yml"),
    )
    conf = ecl_config.Ecl100Config()
    with open("file.yml", "w", encoding="utf-8") as f:
        f.write("this:\n -should\n-be\ninvalid:yaml?")

    monkeypatch.setenv("ECL100_SITE_CONFIG", "file.yml")
    with pytest.raises(ValueError):
        conf = ecl_config.Ecl100Config()

    scalar_exe = "bin/scalar_exe"
    mpi_exe = "bin/mpi_exe"
    mpi_run = "bin/mpi_run"

    os.mkdir("bin")
    for f in ["scalar_exe", "mpi_exe", "mpi_run"]:
        fname = os.path.join("bin", f)
        with open(fname, "w", encoding="utf-8") as fh:
            fh.write("This is an exectable ...")

        os.chmod(fname, stat.S_IEXEC)

    intel_path = "intel"
    monkeypatch.setenv("ENV1", "A")
    monkeypatch.setenv("ENV2", "C")
    mocked_simulator_config = {
        ecl_config.Keys.env: {"LICENSE_SERVER": "license@company.com"},
        ecl_config.Keys.versions: {
            "2015": {
                ecl_config.Keys.scalar: {ecl_config.Keys.executable: scalar_exe},
                ecl_config.Keys.mpi: {
                    ecl_config.Keys.executable: mpi_exe,
                    ecl_config.Keys.mpirun: mpi_run,
                    ecl_config.Keys.env: {
                        "I_MPI_ROOT": "$ENV1:B:$ENV2",
                        "TEST_VAR": "$ENV1.B.$ENV2 $UNKNOWN_VAR",
                        "P4_RSHCOMMAND": "",
                        "LD_LIBRARY_PATH": f"{intel_path}:$LD_LIBRARY_PATH",
                        "PATH": f"{intel_path}/bin64:$PATH",
                    },
                },
            },
            "2016": {
                ecl_config.Keys.scalar: {ecl_config.Keys.executable: "/does/not/exist"},
                ecl_config.Keys.mpi: {
                    ecl_config.Keys.executable: "/does/not/exist",
                    ecl_config.Keys.mpirun: mpi_run,
                },
            },
            "2017": {
                ecl_config.Keys.mpi: {
                    ecl_config.Keys.executable: mpi_exe,
                    ecl_config.Keys.mpirun: "/does/not/exist",
                }
            },
        },
    }

    with open("file.yml", "w", encoding="utf-8") as filehandle:
        filehandle.write(yaml.dump(mocked_simulator_config))

    conf = ecl_config.Ecl100Config()
    # Fails because there is no version 2020
    with pytest.raises(KeyError):
        sim = conf.sim("2020")

    # Fails because the 2016 version points to a not existing executable
    with pytest.raises(OSError):
        sim = conf.sim("2016")

    # Fails because the 2016 mpi version points to a non existing mpi
    # executable
    with pytest.raises(OSError):
        sim = conf.mpi_sim("2016")

    # Fails because the 2017 mpi version mpirun points to a non existing
    # mpi executable
    with pytest.raises(OSError):
        sim = conf.mpi_sim("2017")

    # Fails because the 2017 scalar version is not registered
    with pytest.raises(KeyError):
        sim = conf.sim("2017")

    sim = conf.sim("2015")
    mpi_sim = conf.mpi_sim("2015")

    # Check that global environment has been propagated down.
    assert "LICENSE_SERVER" in mpi_sim.env

    # Check replacement of $ENV_VAR in values.
    assert mpi_sim.env["I_MPI_ROOT"] == "A:B:C"
    assert mpi_sim.env["TEST_VAR"] == "A.B.C $UNKNOWN_VAR"
    assert len(mpi_sim.env) == 1 + 5

    sim = conf.sim("2015")
    assert sim.executable == scalar_exe
    assert sim.mpirun is None

    with pytest.raises(Exception):
        simulators = conf.simulators()

    simulators = conf.simulators(strict=False)
    assert len(simulators) == 2


@pytest.mark.usefixtures("use_tmpdir")
def test_default_version_definitions(monkeypatch):
    os.mkdir("bin")
    scalar_exe = "bin/scalar_exe"
    with open(scalar_exe, "w", encoding="utf-8") as fh:
        fh.write("This is an executable ...")
    os.chmod(scalar_exe, stat.S_IEXEC)

    mock_dict_0 = {
        ecl_config.Keys.versions: {
            "2015": {ecl_config.Keys.scalar: {ecl_config.Keys.executable: scalar_exe}},
            "2016": {ecl_config.Keys.scalar: {ecl_config.Keys.executable: scalar_exe}},
        }
    }

    mock_dict_1 = {
        ecl_config.Keys.default_version: "2015",
        ecl_config.Keys.versions: {
            "2015": {ecl_config.Keys.scalar: {ecl_config.Keys.executable: scalar_exe}},
            "2016": {ecl_config.Keys.scalar: {ecl_config.Keys.executable: scalar_exe}},
        },
    }

    monkeypatch.setenv("ECL100_SITE_CONFIG", os.path.join("file.yml"))
    with open("file.yml", "w", encoding="utf-8") as f:
        f.write(yaml.dump(mock_dict_1))

    conf = ecl_config.Ecl100Config()
    sim = conf.sim()
    assert sim.version == "2015"
    assert "2015" in conf
    assert "xxxx" not in conf
    assert ecl_config.Keys.default in conf
    assert None in conf

    sim = conf.sim("default")
    assert sim.version == "2015"

    with open("file.yml", "w", encoding="utf-8") as filehandle:
        filehandle.write(yaml.dump(mock_dict_0))

    conf = ecl_config.Ecl100Config()
    assert ecl_config.Keys.default not in conf
    assert conf.default_version is None

    with pytest.raises(Exception):
        sim = conf.sim()

    with pytest.raises(Exception):
        sim = conf.sim(ecl_config.Keys.default)
