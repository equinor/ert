import os
from pathlib import Path
from textwrap import dedent


from res.enkf import ErtRunContext, ResConfig, EnKFMain


def test_assert_symlink_deleted(setup_case):
    res_config = setup_case("local/snake_oil_field", "snake_oil.ert")
    ert = EnKFMain(res_config)
    runpath_list = ert.getRunpathList()

    runner = ert.getEnkfSimulationRunner()

    # create directory structure
    model_config = ert.getModelConfig()
    run_context = ErtRunContext.ensemble_experiment(
        ert.getEnkfFsManager().getCurrentFileSystem(),
        [True],
        model_config.getRunpathFormat(),
        model_config.getJobnameFormat(),
        ert.getDataKW(),
        0,
    )
    runner.createRunPath(run_context)

    # replace field file with symlink
    linkpath = "%s/permx.grdcel" % str(runpath_list[0].runpath)
    targetpath = "%s/permx.grdcel.target" % str(runpath_list[0].runpath)
    open(targetpath, "a").close()
    os.remove(linkpath)
    os.symlink(targetpath, linkpath)

    # recreate directory structure
    runner.createRunPath(run_context)

    # ensure field symlink is replaced by file
    assert not os.path.islink(linkpath)


def test_assert_export(use_tmpdir):
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
    runpath_list = ert.getRunpathList()
    assert not os.path.isfile(runpath_list.getExportFile())

    fs_manager = ert.getEnkfFsManager()
    model_config = ert.getModelConfig()
    run_context = ErtRunContext.ensemble_experiment(
        fs_manager.getCurrentFileSystem(),
        [True],
        model_config.getRunpathFormat(),
        model_config.getJobnameFormat(),
        ert.getDataKW(),
        0,
    )

    ert.getEnkfSimulationRunner().createRunPath(run_context)

    assert os.path.isfile(runpath_list.getExportFile())
    assert "test_runpath_list.txt" == os.path.basename(runpath_list.getExportFile())
    assert (
        Path(runpath_list.getExportFile()).read_text("utf-8")
        == f"000  {os.getcwd()}/simulations/realization0  a_name_0  000\n"
    )
