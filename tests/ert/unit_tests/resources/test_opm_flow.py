import os
import platform
import re
import shutil
from pathlib import Path
from subprocess import CalledProcessError

import pytest
import yaml

from tests.ert.utils import SOURCE_DIR

from ._import_from_location import import_from_location

# import ecl_config.py and ecl_run from ert/forward-models/res/script
# package-data path which. These are kept out of the ert package to avoid the
# overhead of importing ert. This is necessary as these may be invoked as a
# subprocess on each realization.

ecl_config = import_from_location(
    "ecl_config",
    SOURCE_DIR / "src/ert/resources/forward_models/res/script/ecl_config.py",
)

ecl_run = import_from_location(
    "ecl_run",
    SOURCE_DIR / "src/ert/resources/forward_models/res/script/ecl_run.py",
)


def locate_flow_binary() -> str:
    """Locate the path for a flow executable.

    Returns the empty string if there is nothing to be found in $PATH."""
    flow_rhel_version = "7"
    if "el8" in platform.release():
        flow_rhel_version = "8"
    candidates = ["flow", f"/project/res/x86_64_RH_{flow_rhel_version}/bin/flowdaily"]
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


@pytest.mark.integration_test
@flow_installed
def test_flow(init_flow_config, source_root):
    shutil.copy(source_root / "test-data/ert/eclipse/SPE1.DATA", "SPE1.DATA")
    shutil.copy(
        source_root / "test-data/ert/eclipse/SPE1_ERROR.DATA", "SPE1_ERROR.DATA"
    )
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


@pytest.mark.integration_test
@flow_installed
def test_flow_with_mpi(init_flow_config, source_root):
    """This only tests that ERT will be able to start flow on a data deck with
    the PARALLEL keyword present. It does not assert anything regarding whether
    MPI-parallelization will get into play."""
    shutil.copy(
        source_root / "test-data/ert/eclipse/SPE1_PARALLEL.DATA", "SPE1_PARALLEL.DATA"
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
def test_running_flow_given_env_variables_with_same_name_as_parent_env_variables_will_overwrite(
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
