import os
from textwrap import dedent

import pytest

from ert._c_wrappers.enkf import EnKFMain, ResConfig
from ert._c_wrappers.enkf.runpaths import Runpaths


@pytest.mark.parametrize(
    "job_format, runpath_format, expected_contents",
    [
        (
            "job%d",
            "/path/to/realization-%d/iteration-%d",
            (
                "003  /path/to/realization-3/iteration-0  job3  000\n"
                "004  /path/to/realization-4/iteration-0  job4  000\n"
                "003  /path/to/realization-3/iteration-1  job3  001\n"
                "004  /path/to/realization-4/iteration-1  job4  001\n"
            ),
        ),
        (
            "job",
            "/path/to/realization-%d/iteration-%d",
            (
                "003  /path/to/realization-3/iteration-0  job  000\n"
                "004  /path/to/realization-4/iteration-0  job  000\n"
                "003  /path/to/realization-3/iteration-1  job  001\n"
                "004  /path/to/realization-4/iteration-1  job  001\n"
            ),
        ),
        (
            "job",
            "/path/to/realization-%d",
            (
                "003  /path/to/realization-3  job  000\n"
                "004  /path/to/realization-4  job  000\n"
                "003  /path/to/realization-3  job  001\n"
                "004  /path/to/realization-4  job  001\n"
            ),
        ),
        (
            "job",
            "/path/to/realization",
            (
                "003  /path/to/realization  job  000\n"
                "004  /path/to/realization  job  000\n"
                "003  /path/to/realization  job  001\n"
                "004  /path/to/realization  job  001\n"
            ),
        ),
    ],
)
def test_runpath_file(tmp_path, job_format, runpath_format, expected_contents):
    runpath_file = tmp_path / "runpath_file"

    assert not runpath_file.exists()

    runpaths = Runpaths(
        job_format,
        runpath_format,
        runpath_file,
    )
    runpaths.write_runpath_list([0, 1], [3, 4])

    assert runpath_file.read_text() == expected_contents


def test_runpath_file_writer_substitution(tmp_path):
    runpath_file = tmp_path / "runpath_file"
    runpaths = Runpaths(
        "<casename>_job",
        "/path/<casename>/ensemble-%d/iteration%d",
        runpath_file,
        lambda x, *_: x.replace("<casename>", "my_case"),
    )

    runpaths.write_runpath_list([1], [1])

    assert (
        runpath_file.read_text()
        == "001  /path/my_case/ensemble-1/iteration1  my_case_job  001\n"
    )


def render_dynamic_values(s, itr, iens, geo_id):
    dynamic_magic_strings = {
        "<GEO_ID>": geo_id,
        "<ITER>": itr,
        "<IENS>": iens,
    }
    for key, val in dynamic_magic_strings.items():
        s = s.replace(key, str(val))

    return s


@pytest.mark.parametrize("itr", [0, 1, 2, 17])
def test_write_snakeoil_runpath_file(snake_oil_case, itr):
    ert = snake_oil_case
    fs_manager = ert.getEnkfFsManager()
    sim_fs = fs_manager.getFileSystem("sim_fs")

    num_realizations = 25
    mask = [True] * num_realizations
    mask[13] = False
    runpath_fmt = (
        "simulations/<GEO_ID>/realization-%d/iter-%d/"
        "magic-real-<IENS>/magic-iter-<ITER>"
    )
    jobname_fmt = "SNAKE_OIL_%d"
    ert.runpaths._runpath_format = runpath_fmt
    ert.runpaths._job_name_format = jobname_fmt

    for i in range(num_realizations):
        ert.set_geo_id(str(10 * i), i, itr)

    run_context = ert.create_ensemble_experiment_run_context(
        source_filesystem=sim_fs,
        active_mask=mask,
        iteration=itr,
    )

    ert.createRunPath(run_context)

    for i, _ in enumerate(run_context):
        if not mask[i]:
            continue

        assert os.path.isdir(f"simulations/{10*i}")

    runpath_list_path = ".ert_runpath_list"
    assert os.path.isfile(runpath_list_path)

    exp_runpaths = [
        runpath_fmt.replace("<ITER>", str(itr))
        .replace("<IENS>", str(iens))
        .replace("<GEO_ID>", str(10 * iens))
        % (iens, itr)
        for iens, _ in enumerate(run_context)
        if mask[iens]
    ]
    exp_runpaths = list(map(os.path.realpath, exp_runpaths))

    with open(runpath_list_path, "r") as f:
        dumped_runpaths = list(zip(*[line.split() for line in f.readlines()]))[1]

    assert list(exp_runpaths) == list(dumped_runpaths)


@pytest.mark.usefixtures("use_tmpdir")
def test_assert_export():
    # Write a minimal config file with env
    with open("config_file.ert", "w") as fout:
        fout.write(
            dedent(
                """
        NUM_REALIZATIONS 1
        JOBNAME a_name_%d
        RUNPATH_FILE directory/test_runpath_list.txt
        """
            )
        )
    res_config = ResConfig("config_file.ert")
    ert = EnKFMain(res_config)
    runpath_list_file = ert.runpath_list_filename
    assert not runpath_list_file.exists()

    run_context = ert.create_ensemble_experiment_run_context(
        iteration=0,
    )

    ert.createRunPath(run_context)

    assert runpath_list_file.exists()
    assert "test_runpath_list.txt" == runpath_list_file.name
    assert (
        runpath_list_file.read_text("utf-8")
        == f"000  {os.getcwd()}/simulations/realization0  a_name_0  000\n"
    )
