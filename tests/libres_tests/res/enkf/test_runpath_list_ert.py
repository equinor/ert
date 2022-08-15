import os
from textwrap import dedent

import pytest
from ert._c_wrappers.enkf import EnKFMain, ResConfig


def test_assert_symlink_deleted(setup_case):
    res_config = setup_case("local/snake_oil_field", "snake_oil.ert")
    ert = EnKFMain(res_config)
    runner = ert

    # create directory structure
    run_context = ert.create_ensemble_experiment_run_context(iteration=0)
    runner.createRunPath(run_context)

    # replace field file with symlink
    linkpath = f"{run_context[0].runpath}/permx.grdcel"
    targetpath = f"{run_context[0].runpath}/permx.grdcel.target"
    open(targetpath, "a").close()
    os.remove(linkpath)
    os.symlink(targetpath, linkpath)

    # recreate directory structure
    runner.createRunPath(run_context)

    # ensure field symlink is replaced by file
    assert not os.path.islink(linkpath)


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
