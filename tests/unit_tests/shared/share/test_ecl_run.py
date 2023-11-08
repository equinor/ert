import os
import re
import shutil
import stat
import sys
from pathlib import Path
from subprocess import CalledProcessError
from textwrap import dedent

import pytest
import yaml
from resdata.summary import Summary

from tests.utils import SOURCE_DIR

from ._import_from_location import import_from_location

# import ecl_config.py and ecl_run from ert/forward-models/res/script
# package-data path which. These are kept out of the ert package to avoid the
# overhead of importing ert. This is necessary as these may be invoked as a
# subprocess on each realization.


ecl_config = import_from_location(
    "ecl_config",
    SOURCE_DIR / "src/ert/shared/share/ert/forward-models/res/script/ecl_config.py",
)

ecl_run = import_from_location(
    "ecl_run",
    SOURCE_DIR / "src/ert/shared/share/ert/forward-models/res/script/ecl_run.py",
)


def locate_flow_binary() -> str:
    """Locate the path for a flow executable.

    Returns the empty string if there is nothing to be found in $PATH."""
    candidates = ["flow", "/project/res/x86_64_RH_7/bin/flowdaily"]
    for candidate in candidates:
        foundpath = shutil.which(candidate)
        if foundpath is not None:
            return foundpath
    return ""


flow_installed = pytest.mark.skipif(
    not locate_flow_binary(), reason="Requires flow to be installed in $PATH"
)


def find_version(output):
    return re.search(r"flow\s*([\d.]+)", output).group(1)


@pytest.fixture(name="init_ecl100_config")
def fixture_init_ecl100_config(monkeypatch, tmpdir):
    ecl19_prefix = "/prog/res/ecl/grid/2019.3/bin/linux_x86_64/"
    ecl22_prefix = "/prog/ecl/grid/2022.4/bin/linux_x86_64/"
    mpi_prefix = "/prog/ecl/grid/tools/linux_x86_64/intel/mpi/2021.4.0/"
    mpi_libs = mpi_prefix + "/lib/release" + ":" + mpi_prefix + "/lib"
    mpi_bins = mpi_prefix + "/bin"
    conf = {
        "env": {
            "F_UFMTENDIAN": "big",
            "LM_LICENSE_FILE": "7321@eclipse-lic-no.statoil.no",
            "ARCH": "x86_64",
        },
        "versions": {
            "2022.4": {
                "scalar": {"executable": ecl22_prefix + "eclipse.exe"},
                "mpi": {
                    "executable": ecl22_prefix + "eclipse_ilmpi.exe",
                    "mpirun": mpi_prefix + "bin/mpirun",
                    "env": {
                        "I_MPI_ROOT": mpi_prefix,
                        "P4_RSHCOMMAND": "ssh",
                        "LD_LIBRARY_PATH": mpi_libs + ":$LD_LIBRARY_PATH",
                        "PATH": mpi_bins + ":$PATH",
                    },
                },
            },
            "2019.3": {
                "scalar": {"executable": ecl19_prefix + "eclipse.exe"},
                "mpi": {
                    "executable": ecl19_prefix + "eclipse_ilmpi.exe",
                    "mpirun": mpi_prefix + "bin/mpirun",
                    "env": {
                        "I_MPI_ROOT": mpi_prefix,
                        "P4_RSHCOMMAND": "ssh",
                        "LD_LIBRARY_PATH": mpi_libs + ":$LD_LIBRARY_PATH",
                        "PATH": mpi_bins + ":$PATH",
                    },
                },
            },
        },
    }
    with tmpdir.as_cwd():
        with open("ecl100_config.yml", "w", encoding="utf-8") as filehandle:
            filehandle.write(yaml.dump(conf))
        monkeypatch.setenv("ECL100_SITE_CONFIG", "ecl100_config.yml")
        yield


@pytest.fixture(name="init_flow_config")
def fixture_init_flow_config(monkeypatch, tmpdir):
    conf = {
        "default_version": "default",
        "versions": {"default": {"scalar": {"executable": locate_flow_binary()}}},
    }
    with tmpdir.as_cwd():
        Path("flow_config.yml").write_text(yaml.dump(conf), encoding="utf-8")
        monkeypatch.setenv("FLOW_SITE_CONFIG", "flow_config.yml")
        yield


def test_ecl_run_make_LSB_MCPU_machine_list():
    assert ecl_run.make_LSB_MCPU_machine_list("host1 4 host2 4") == [
        "host1",
        "host1",
        "host1",
        "host1",
        "host2",
        "host2",
        "host2",
        "host2",
    ]


@pytest.mark.usefixtures("use_tmpdir")
def test_mocked_simulator_configuration(monkeypatch):
    conf = {
        "versions": {
            "mocked": {
                "scalar": {"executable": "bin/scalar_exe"},
                "mpi": {"executable": "bin/mpi_exe", "mpirun": "bin/mpirun"},
            }
        }
    }
    with open("ecl100_config.yml", "w", encoding="utf-8") as filehandle:
        filehandle.write(yaml.dump(conf))

    os.mkdir("bin")
    monkeypatch.setenv("ECL100_SITE_CONFIG", "ecl100_config.yml")
    for filename in ["scalar_exe", "mpi_exe", "mpirun"]:
        fname = os.path.join("bin", filename)
        with open(fname, "w", encoding="utf-8") as filehandle:
            filehandle.write("This is an executable ...")

        os.chmod(fname, stat.S_IEXEC)

    with open("ECLIPSE.DATA", "w", encoding="utf-8") as filehandle:
        filehandle.write("Mock eclipse data file")

    econfig = ecl_config.Ecl100Config()
    sim = econfig.sim("mocked")
    mpi_sim = econfig.mpi_sim("mocked")
    erun = ecl_run.EclRun("ECLIPSE.DATA", sim)
    assert erun.runPath() == os.getcwd()

    os.mkdir("path")
    with open("path/ECLIPSE.DATA", "w", encoding="utf-8") as filehandle:
        filehandle.write("Mock eclipse data file")

    erun = ecl_run.EclRun("path/ECLIPSE.DATA", sim)
    assert erun.runPath() == os.path.join(os.getcwd(), "path")
    assert erun.baseName() == "ECLIPSE"
    assert erun.numCpu() == 1

    # invalid number of CPU
    with pytest.raises(ValueError):
        ecl_run.EclRun("path/ECLIPSE.DATA", sim, num_cpu="xxx")

    erun = ecl_run.EclRun("path/ECLIPSE.DATA", mpi_sim, num_cpu="10")
    assert erun.numCpu() == 10

    # Missing datafile
    with pytest.raises(IOError):
        ecl_run.EclRun("DOES/NOT/EXIST", mpi_sim, num_cpu="10")


@flow_installed
def test_flow(init_flow_config, source_root):
    shutil.copy(source_root / "test-data/eclipse/SPE1.DATA", "SPE1.DATA")
    shutil.copy(source_root / "test-data/eclipse/SPE1_ERROR.DATA", "SPE1_ERROR.DATA")
    flow_config = ecl_config.FlowConfig()
    sim = flow_config.sim()
    flow_run = ecl_run.EclRun("SPE1.DATA", sim)
    flow_run.runEclipse()

    ecl_run.run(flow_config, ["SPE1.DATA"])

    flow_run = ecl_run.EclRun("SPE1_ERROR.DATA", sim)
    with pytest.raises(CalledProcessError, match="returned non-zero exit status 1"):
        flow_run.runEclipse()

    ecl_run.run(flow_config, ["SPE1_ERROR.DATA", "--ignore-errors"])

    # Invalid version
    with pytest.raises(KeyError):
        ecl_run.run(flow_config, ["SPE1.DATA", "--version=no/such/version"])


@flow_installed
def test_flow_with_mpi(init_flow_config, source_root):
    """This only tests that ERT will be able to start flow on a data deck with
    the PARALLEL keyword present. It does not assert anything regarding whether
    MPI-parallelization will get into play."""
    shutil.copy(
        source_root / "test-data/eclipse/SPE1_PARALLEL.DATA", "SPE1_PARALLEL.DATA"
    )
    flow_config = ecl_config.FlowConfig()
    sim = flow_config.sim()
    flow_run = ecl_run.EclRun("SPE1_PARALLEL.DATA", sim)
    flow_run.runEclipse()


@pytest.mark.usefixtures("use_tmpdir")
def test_running_flow_given_env_config_can_still_read_parent_env(monkeypatch):
    version = "1111.11"

    # create a script that prints env vars ENV1 and ENV2 to a file
    with open("mocked_flow", "w", encoding="utf-8") as f:
        f.write("#!/bin/sh\n")
        f.write("echo $ENV1 > out.txt\n")
        f.write("echo $ENV2 >> out.txt\n")
    executable = os.path.join(os.getcwd(), "mocked_flow")
    os.chmod(executable, 0o777)

    # create a flow_config.yml with environment extension ENV2
    conf = {
        "default_version": version,
        "versions": {
            version: {
                "scalar": {"executable": executable, "env": {"ENV2": "VAL2"}},
            }
        },
    }

    with open("flow_config.yml", "w", encoding="utf-8") as filehandle:
        filehandle.write(yaml.dump(conf))

    # set the environment variable ENV1
    monkeypatch.setenv("ENV1", "VAL1")
    monkeypatch.setenv("FLOW_SITE_CONFIG", "flow_config.yml")

    with open("DUMMY.DATA", "w", encoding="utf-8") as filehandle:
        filehandle.write("dummy")

    with open("DUMMY.PRT", "w", encoding="utf-8") as filehandle:
        filehandle.write("Errors 0\n")
        filehandle.write("Bugs 0\n")

    # run the script
    flow_config = ecl_config.FlowConfig()
    sim = flow_config.sim()
    flow_run = ecl_run.EclRun("DUMMY.DATA", sim)
    flow_run.runEclipse()

    # assert that the script was able to read both the variables correctly
    with open("out.txt", encoding="utf-8") as filehandle:
        lines = filehandle.readlines()

    assert lines == ["VAL1\n", "VAL2\n"]


@pytest.mark.usefixtures("use_tmpdir")
def test_running_flow_given_no_env_config_can_still_read_parent_env(monkeypatch):
    version = "1111.11"

    # create a script that prints env vars ENV1 and ENV2 to a file
    with open("flow", "w", encoding="utf-8") as f:
        f.write("#!/bin/sh\n")
        f.write("echo $ENV1 > out.txt\n")
        f.write("echo $ENV2 >> out.txt\n")
    executable = os.path.join(os.getcwd(), "flow")
    os.chmod(executable, 0o777)

    # create a flow_config.yml with environment extension ENV2
    conf = {
        "default_version": version,
        "versions": {
            version: {
                "scalar": {"executable": executable},
            }
        },
    }

    with open("flow_config.yml", "w", encoding="utf-8") as filehandle:
        filehandle.write(yaml.dump(conf))

    # set the environment variable ENV1
    monkeypatch.setenv("ENV1", "VAL1")
    monkeypatch.setenv("ENV2", "VAL2")
    monkeypatch.setenv("FLOW_SITE_CONFIG", "flow_config.yml")

    with open("DUMMY.DATA", "w", encoding="utf-8") as filehandle:
        filehandle.write("dummy")

    with open("DUMMY.PRT", "w", encoding="utf-8") as filehandle:
        filehandle.write("Errors 0\n")
        filehandle.write("Bugs 0\n")

    # run the script
    flow_config = ecl_config.FlowConfig()
    sim = flow_config.sim()
    flow_run = ecl_run.EclRun("DUMMY.DATA", sim)
    flow_run.runEclipse()

    # assert that the script was able to read both the variables correctly
    with open("out.txt", encoding="utf-8") as filehandle:
        lines = filehandle.readlines()

    assert lines == ["VAL1\n", "VAL2\n"]


@pytest.mark.usefixtures("use_tmpdir")
def test_running_flow_given_env_variables_with_same_name_as_parent_env_variables_will_overwrite(  # noqa
    monkeypatch,
):
    version = "1111.11"

    # create a script that prints env vars ENV1 and ENV2 to a file
    with open("flow", "w", encoding="utf-8") as filehandle:
        filehandle.write("#!/bin/sh\n")
        filehandle.write("echo $ENV1 > out.txt\n")
        filehandle.write("echo $ENV2 >> out.txt\n")
    executable = os.path.join(os.getcwd(), "flow")
    os.chmod(executable, 0o777)

    # create a flow_config.yml with environment extension ENV2
    conf = {
        "default_version": version,
        "versions": {
            version: {
                "scalar": {
                    "executable": executable,
                    "env": {"ENV1": "OVERWRITTEN1", "ENV2": "OVERWRITTEN2"},
                },
            }
        },
    }

    with open("flow_config.yml", "w", encoding="utf-8") as filehandle:
        filehandle.write(yaml.dump(conf))

    # set the environment variable ENV1
    monkeypatch.setenv("ENV1", "VAL1")
    monkeypatch.setenv("ENV2", "VAL2")
    monkeypatch.setenv("FLOW_SITE_CONFIG", "flow_config.yml")

    with open("DUMMY.DATA", "w", encoding="utf-8") as filehandle:
        filehandle.write("dummy")

    with open("DUMMY.PRT", "w", encoding="utf-8") as filehandle:
        filehandle.write("Errors 0\n")
        filehandle.write("Bugs 0\n")

    # run the script
    flow_config = ecl_config.FlowConfig()
    sim = flow_config.sim()
    flow_run = ecl_run.EclRun("DUMMY.DATA", sim)
    flow_run.runEclipse()

    # assert that the script was able to read both the variables correctly
    with open("out.txt", encoding="utf-8") as filehandle:
        lines = filehandle.readlines()

    assert lines == ["OVERWRITTEN1\n", "OVERWRITTEN2\n"]


@pytest.mark.requires_eclipse
def test_run(init_ecl100_config, source_root):
    shutil.copy(
        source_root / "test-data/eclipse/SPE1.DATA",
        "SPE1.DATA",
    )
    econfig = ecl_config.Ecl100Config()
    sim = econfig.sim("2019.3")
    erun = ecl_run.EclRun("SPE1.DATA", sim)
    erun.runEclipse()

    ok_filepath = os.path.join(erun.runPath(), f"{erun.baseName()}.OK")

    # PRT is an Eclipse-specific extension for its "print-file",
    # essentially a log-file.
    prt_filepath = os.path.join(erun.runPath(), f"{erun.baseName()}.PRT")

    assert os.path.isfile(ok_filepath)
    assert os.path.isfile(prt_filepath)
    assert os.path.getsize(prt_filepath) > 0

    assert not erun.parseErrors()


@pytest.mark.requires_eclipse
def test_run_nonzero_exit_code(init_ecl100_config, source_root):
    """Monkey patching the erun to use an executable which will fail with
    exit(1); don't think Eclipse actually fails with exit(1) - but let us at
    least be prepared when/if it does."""
    Path("FOO.DATA").write_text("", encoding="utf-8")
    econfig = ecl_config.Ecl100Config()
    sim = econfig.sim("2019.3")
    erun = ecl_run.EclRun("FOO.DATA", sim)
    erun.sim.executable = source_root / "tests/unit_tests/shared/share/ecl_run_fail"

    with pytest.raises(CalledProcessError):
        erun.runEclipse()


@pytest.mark.requires_eclipse
def test_run_api(init_ecl100_config, source_root):
    shutil.copy(
        source_root / "test-data/eclipse/SPE1.DATA",
        "SPE1.DATA",
    )
    econfig = ecl_config.Ecl100Config()
    ecl_run.run(econfig, ["SPE1.DATA", "--version=2019.3"])

    assert os.path.isfile("SPE1.DATA")


@pytest.mark.requires_eclipse
def test_failed_run(init_ecl100_config, source_root):
    shutil.copy(
        source_root / "test-data/eclipse/SPE1_ERROR.DATA",
        "SPE1_ERROR.DATA",
    )
    econfig = ecl_config.Ecl100Config()
    sim = econfig.sim("2019.3")
    erun = ecl_run.EclRun("SPE1_ERROR", sim)
    with pytest.raises(RuntimeError, match="ERROR  AT TIME        0.0   DAYS "):
        erun.runEclipse()


@pytest.mark.requires_eclipse
def test_failed_run_OK(init_ecl100_config, source_root):
    shutil.copy(
        source_root / "test-data/eclipse/SPE1_ERROR.DATA",
        "SPE1_ERROR.DATA",
    )
    econfig = ecl_config.Ecl100Config()
    ecl_run.run(econfig, ["SPE1_ERROR", "--version=2022.4", "--ignore-errors"])

    # Monkey patching the ecl_run to use an executable which will fail with exit(1),
    # in the nocheck mode that should also be OK.
    sim = econfig.sim("2019.3")
    erun = ecl_run.EclRun("SPE1_ERROR", sim, check_status=False)
    erun.sim.executable = source_root / "tests/unit_tests/shared/share/ecl_run_fail"
    erun.runEclipse()


@pytest.mark.requires_eclipse
def test_mpi_run(init_ecl100_config, source_root):
    shutil.copy(
        source_root / "test-data/eclipse/SPE1_PARALLEL.DATA",
        "SPE1_PARALLEL.DATA",
    )
    econfig = ecl_config.Ecl100Config()
    ecl_run.run(econfig, ["SPE1_PARALLEL.DATA", "--version=2022.4", "--num-cpu=2"])
    assert os.path.isfile("SPE1_PARALLEL.PRT")
    assert os.path.getsize("SPE1_PARALLEL.PRT") > 0


@pytest.mark.requires_eclipse
def test_summary_block(init_ecl100_config, source_root):
    shutil.copy(
        source_root / "test-data/eclipse/SPE1.DATA",
        "SPE1.DATA",
    )
    econfig = ecl_config.Ecl100Config()
    sim = econfig.sim("2022.4")
    erun = ecl_run.EclRun("SPE1.DATA", sim)
    ret_value = erun.summary_block()
    assert ret_value is None

    erun.runEclipse()
    assert isinstance(erun.summary_block(), Summary)


@pytest.mark.requires_eclipse
def test_error_parse(init_ecl100_config, source_root):
    shutil.copy(
        source_root / "test-data/eclipse/SPE1.DATA",
        "SPE1.DATA",
    )
    prt_file = source_root / "test-data/eclipse/parse/ERROR.PRT"
    shutil.copy(prt_file, "SPE1.PRT")

    econfig = ecl_config.Ecl100Config()
    sim = econfig.sim("2019.3")
    erun = ecl_run.EclRun("SPE1.DATA", sim)

    error0 = (
        " @--  ERROR  AT TIME        0.0   DAYS    ( 1-JAN-0):\n"
        " @           UNABLE TO OPEN INCLUDED FILE                                    \n"  # noqa
        " @           /private/joaho/ERT/git/Gurbat/XXexample_grid_sim.GRDECL         \n"  # noqa
        " @           SYSTEM ERROR CODE IS       29                                   "
    )

    error1 = (
        " @--  ERROR  AT TIME        0.0   DAYS    ( 1-JAN-0):\n"
        " @           INCLUDE FILES MISSING.                                          "
    )

    assert erun.parseErrors() == [error0, error1]


def test_slurm_env_parsing():
    host_list = ecl_run.make_SLURM_machine_list("ws", "2")
    assert host_list == ["ws", "ws"]

    host_list = ecl_run.make_SLURM_machine_list("ws1,ws2", "2,3")
    assert host_list == ["ws1", "ws1", "ws2", "ws2", "ws2"]

    host_list = ecl_run.make_SLURM_machine_list("ws[1-3]", "1,2,3")
    assert host_list == ["ws1", "ws2", "ws2", "ws3", "ws3", "ws3"]

    host_list = ecl_run.make_SLURM_machine_list("ws[1,3]", "1,3")
    assert host_list == ["ws1", "ws3", "ws3", "ws3"]

    host_list = ecl_run.make_SLURM_machine_list("ws[1-3,6-8]", "1,2,3,1,2,3")
    assert host_list == [
        "ws1",
        "ws2",
        "ws2",
        "ws3",
        "ws3",
        "ws3",
        "ws6",
        "ws7",
        "ws7",
        "ws8",
        "ws8",
        "ws8",
    ]

    host_list = ecl_run.make_SLURM_machine_list("ws[1-3,6-8]", "2(x2),3,1,2(x2)")
    assert host_list == [
        "ws1",
        "ws1",
        "ws2",
        "ws2",
        "ws3",
        "ws3",
        "ws3",
        "ws6",
        "ws7",
        "ws7",
        "ws8",
        "ws8",
    ]

    host_list = ecl_run.make_SLURM_machine_list("ws[1-3,6],ws[7-8]", "2(x2),3,1,2(x2)")
    assert host_list == [
        "ws1",
        "ws1",
        "ws2",
        "ws2",
        "ws3",
        "ws3",
        "ws3",
        "ws6",
        "ws7",
        "ws7",
        "ws8",
        "ws8",
    ]


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="Flaky bash mock on mac")
def test_ecl100_retries_once_on_license_failure(tmp_path, monkeypatch):
    mock_eclipse_path = tmp_path / "mock_eclipse100"
    with open(tmp_path / "mock_config.yaml", "w", encoding="utf-8") as fp:
        yaml.dump(
            {
                "versions": {
                    "2015.2": {"scalar": {"executable": str(mock_eclipse_path)}}
                }
            },
            fp,
        )

    case_path = tmp_path / "CASE.DATA"
    case_path.write_text("", encoding="utf-8")
    mock_eclipse_path.write_text(
        dedent(
            """\
        #!/usr/bin/bash
        echo 'Errors 1
        Bugs 0
         @--  ERROR  AT TIME        0.0   DAYS    ( 1-JAN-2000):
         @       LICENSE FAILURE: ERROR NUMBER IS -33' > CASE.PRT
        echo 'Called mock' >> mock_log
        """
        ),
        encoding="utf-8",
    )
    mock_eclipse_path.chmod(
        stat.S_IEXEC | stat.S_IWUSR | mock_eclipse_path.stat().st_mode
    )
    monkeypatch.setenv("ECL100_SITE_CONFIG", str(tmp_path / "mock_config.yaml"))
    econfig = ecl_config.Ecl100Config()
    sim = econfig.sim("2015.2")
    erun = ecl_run.EclRun(str(case_path), sim)
    erun.LICENSE_FAILURE_SLEEP_SECONDS = 1

    with pytest.raises(RuntimeError, match="LICENSE FAILURE"):
        erun.runEclipse()
    max_attempts = 2
    assert (tmp_path / "mock_log").read_text() == "Called mock\n" * max_attempts
