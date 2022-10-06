import os
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import pytest

from ert._c_wrappers.enkf import EnKFMain, ResConfig


@pytest.mark.usefixtures("use_tmpdir")
def test_that_run_template_replace_symlink_does_not_write_to_source():
    """This test is meant to test that we can have a symlinked file in the
    run path before we do replacement on a target file with the same name,
    the described behavior is:
    >     If the target_file already exists as a symbolic link, the
    >     symbolic link will be removed prior to creating the instance,
    >     ensuring that a remote file is not updated.
    it also has the side effect of testing that we are able to create the
    run path although the expected folders are already present
    """
    config_text = dedent(
        """
        NUM_REALIZATIONS 1
        JOBNAME my_case%d
        RUN_TEMPLATE template.tmpl result.txt
        """
    )
    Path("template.tmpl").write_text("I want to replace: <IENS>")
    Path("config.ert").write_text(config_text)
    res_config = ResConfig("config.ert")
    ert = EnKFMain(res_config)
    run_context = ert.create_ensemble_experiment_run_context(
        iteration=0, active_mask=[True]
    )
    run_path = Path(run_context.paths[0])
    os.makedirs(run_path)
    # Write a file that will be symlinked into the run run path with the
    # same name as the target_file
    Path("start.txt").write_text("I dont want to replace in this file")
    os.symlink("start.txt", run_path / "result.txt")
    ert.createRunPath(run_context)
    assert (run_path / "result.txt").read_text() == "I want to replace: 0"
    # Check that the source of the symlinked file is not updated
    assert Path("start.txt").read_text() == "I dont want to replace in this file"


@pytest.mark.usefixtures("use_tmpdir")
def test_run_template_replace_in_file_with_custom_define():
    """
    This test checks that we are able to magically replace custom magic
    strings using the DEFINE keyword
    """
    config_text = dedent(
        """
        NUM_REALIZATIONS 1
        JOBNAME my_case%d
        DEFINE <MY_VAR> my_custom_variable
        RUN_TEMPLATE template.tmpl result.txt
        """
    )
    Path("template.tmpl").write_text("I WANT TO REPLACE:<MY_VAR>")
    Path("config.ert").write_text(config_text)

    res_config = ResConfig("config.ert")
    ert = EnKFMain(res_config)
    run_context = ert.create_ensemble_experiment_run_context(
        iteration=0, active_mask=[True]
    )
    ert.createRunPath(run_context)
    assert (
        Path(run_context.paths[0]) / "result.txt"
    ).read_text() == "I WANT TO REPLACE:my_custom_variable"


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "key, expected",
    [
        ("<DATE>", datetime.date(datetime.today()).isoformat()),
        ("<NUM_CPU>", "1"),
        ("<CONFIG_FILE_BASE>", "config"),
        ("<CONFIG_FILE>", "config.ert"),
        ("<ERT-CASE>", "default"),
        ("<ERTCASE>", "default"),
        ("<ECL_BASE>", "my_case0"),
        ("<ECLBASE>", "my_case0"),
        ("<IENS>", "0"),
        ("<ITER>", "0"),
    ],
)
def test_run_template_replace_in_file(key, expected):
    config_text = dedent(
        """
        NUM_REALIZATIONS 1
        JOBNAME my_case%d
        RUN_TEMPLATE template.tmpl result.txt
        """
    )
    Path("template.tmpl").write_text(f"I WANT TO REPLACE:{key}")
    Path("config.ert").write_text(config_text)

    res_config = ResConfig("config.ert")
    ert = EnKFMain(res_config)
    run_context = ert.create_ensemble_experiment_run_context(
        iteration=0, active_mask=[True]
    )
    ert.createRunPath(run_context)
    assert (
        Path(run_context.paths[0]) / "result.txt"
    ).read_text() == f"I WANT TO REPLACE:{expected}"


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "ecl_base, expected_file",
    (
        ("MY_ECL_BASE", "MY_ECL_BASE.DATA"),
        ("relative/path/MY_ECL_BASE", "relative/path/MY_ECL_BASE.DATA"),
        ("MY_ECL_BASE%d", "MY_ECL_BASE0.DATA"),
        ("MY_ECL_BASE<IENS>", "MY_ECL_BASE0.DATA"),
    ),
)
def test_run_template_replace_in_ecl(ecl_base, expected_file):
    config_text = dedent(
        f"""
        NUM_REALIZATIONS 1
        ECLBASE {ecl_base}
        RUN_TEMPLATE BASE_ECL_FILE.DATA <ECLBASE>.DATA
        """
    )
    Path("BASE_ECL_FILE.DATA").write_text("I WANT TO REPLACE:<NUM_CPU>")
    Path("config.ert").write_text(config_text)

    res_config = ResConfig("config.ert")
    ert = EnKFMain(res_config)
    run_context = ert.create_ensemble_experiment_run_context(
        iteration=0, active_mask=[True]
    )
    ert.createRunPath(run_context)
    assert (
        Path(run_context.paths[0]) / expected_file
    ).read_text() == "I WANT TO REPLACE:1"


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "key, expected",
    [
        ("<DATE>", datetime.date(datetime.today()).isoformat()),
        ("<NUM_CPU>", "1"),
        ("<CONFIG_FILE_BASE>", "config"),
        ("<CONFIG_FILE>", "config.ert"),
        ("<ERT-CASE>", "default"),
        ("<ERTCASE>", "default"),
        ("<ECL_BASE>", "ECL_CASE0"),
        ("<ECLBASE>", "ECL_CASE0"),
        ("<IENS>", "0"),
        ("<ITER>", "0"),
    ],
)
def test_run_template_replace_in_ecl_data_file(key, expected):
    """
    This test that we copy the DATA_FILE into the runpath,
    do substitutions and rename it from the DATA_FILE name
    to ECLBASE
    """
    config_text = dedent(
        """
        NUM_REALIZATIONS 1
        ECLBASE ECL_CASE%d
        DATA_FILE MY_DATA_FILE.DATA
        """
    )
    Path("MY_DATA_FILE.DATA").write_text(f"I WANT TO REPLACE:{key}")
    Path("config.ert").write_text(config_text)

    res_config = ResConfig("config.ert")
    ert = EnKFMain(res_config)
    run_context = ert.create_ensemble_experiment_run_context(
        iteration=0, active_mask=[True]
    )
    ert.createRunPath(run_context)
    assert (
        Path(run_context.paths[0]) / "ECL_CASE0.DATA"
    ).read_text() == f"I WANT TO REPLACE:{expected}"


@pytest.mark.usefixtures("use_tmpdir")
def test_run_template_replace_in_file_name():
    """
    This test checks that we are able to magically replace custom magic
    strings using the DEFINE keyword
    """
    config_text = dedent(
        """
        NUM_REALIZATIONS 1
        JOBNAME my_case%d
        DEFINE <MY_FILE_NAME> result.txt
        RUN_TEMPLATE template.tmpl <MY_FILE_NAME>
        """
    )
    Path("template.tmpl").write_text("Not important, name of the file is important")
    Path("config.ert").write_text(config_text)

    res_config = ResConfig("config.ert")
    ert = EnKFMain(res_config)
    run_context = ert.create_ensemble_experiment_run_context(
        iteration=0, active_mask=[True]
    )
    ert.createRunPath(run_context)
    assert (
        Path(run_context.paths[0]) / "result.txt"
    ).read_text() == "Not important, name of the file is important"


@pytest.mark.usefixtures("use_tmpdir")
def test_that_creating_runpath_makes_initialized_fs():
    """
    This checks that creating the run path initializes the selected case,
    for that parameters are needed, so add a simple GEN_KW.
    """
    config_text = dedent(
        """
        NUM_REALIZATIONS 1
        JOBNAME my_case%d
        GEN_KW KW_NAME template.txt kw.txt prior.txt
        """
    )
    Path("template.tmpl").write_text("Not important, name of the file is important")
    Path("config.ert").write_text(config_text)
    with open("template.txt", "w") as fh:
        fh.writelines("MY_KEYWORD <MY_KEYWORD>")
    with open("prior.txt", "w") as fh:
        fh.writelines("MY_KEYWORD NORMAL 0 1")
    res_config = ResConfig("config.ert")
    ert = EnKFMain(res_config)
    run_context = ert.create_ensemble_experiment_run_context(
        iteration=0, active_mask=[True]
    )
    assert not ert.isCaseInitialized("default")
    ert.createRunPath(run_context)
    assert ert.isCaseInitialized("default")


@pytest.mark.parametrize(
    "eclipse_data, expected_cpus",
    [
        ("PARALLEL 4 /", 4),
        pytest.param(
            dedent(
                """
            SLAVES
            -- comment
            -- comment with slash / "
            'upper' 'base' '*' 'data_file' /
            'lower' 'base' '*' 'data_file_lower' /
            /"""
            ),
            3,
            id=(
                "This is strictly not the correct way to parse this, "
                "as the slave nodes could run on multiple cores"
            ),
        ),
    ],
)
@pytest.mark.usefixtures("use_tmpdir")
def test_that_data_file_sets_num_cpu(eclipse_data, expected_cpus):
    config_text = dedent(
        """
        NUM_REALIZATIONS 1
        DATA_FILE MY_DATA_FILE.DATA
        """
    )
    Path("MY_DATA_FILE.DATA").write_text(eclipse_data)
    Path("config.ert").write_text(config_text)

    res_config = ResConfig("config.ert")
    ert = EnKFMain(res_config)
    assert int(ert.substituter.get_substitutions(0, 0)["<NUM_CPU>"]) == expected_cpus
