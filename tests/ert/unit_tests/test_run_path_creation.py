import os
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import orjson
import pytest

from ert.config import ConfigValidationError, ErtConfig
from ert.enkf_main import create_run_path, sample_prior
from ert.run_arg import create_run_arguments
from ert.runpaths import Runpaths
from ert.storage import Storage


@pytest.mark.usefixtures("use_tmpdir")
def test_that_run_template_replace_symlink_does_not_write_to_source(
    prior_ensemble, run_args, run_paths
):
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
    Path("template.tmpl").write_text("I want to replace: <IENS>", encoding="utf-8")
    Path("config.ert").write_text(config_text, encoding="utf-8")
    ert_config = ErtConfig.from_file("config.ert")
    run_arg = run_args(ert_config, prior_ensemble)
    run_path = Path(run_arg[0].runpath)
    os.makedirs(run_path)
    # Write a file that will be symlinked into the run run path with the
    # same name as the target_file
    Path("start.txt").write_text(
        "I dont want to replace in this file", encoding="utf-8"
    )
    os.symlink("start.txt", run_path / "result.txt")
    create_run_path(
        run_args=run_arg,
        ensemble=prior_ensemble,
        user_config_file=ert_config.user_config_file,
        env_vars=ert_config.env_vars,
        forward_model_steps=ert_config.forward_model_steps,
        substitutions=ert_config.substitutions,
        templates=ert_config.ert_templates,
        model_config=ert_config.model_config,
        runpaths=run_paths(ert_config),
    )
    assert (run_path / "result.txt").read_text(
        encoding="utf-8"
    ) == "I want to replace: 0"
    # Check that the source of the symlinked file is not updated
    assert (
        Path("start.txt").read_text(encoding="utf-8")
        == "I dont want to replace in this file"
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_run_template_replace_in_file_with_custom_define(
    prior_ensemble, run_args, run_paths
):
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
    Path("template.tmpl").write_text("I WANT TO REPLACE:<MY_VAR>", encoding="utf-8")
    Path("config.ert").write_text(config_text, encoding="utf-8")

    ert_config = ErtConfig.from_file("config.ert")
    run_arg = run_args(ert_config, prior_ensemble)
    create_run_path(
        run_args=run_arg,
        ensemble=prior_ensemble,
        user_config_file=ert_config.user_config_file,
        env_vars=ert_config.env_vars,
        forward_model_steps=ert_config.forward_model_steps,
        substitutions=ert_config.substitutions,
        templates=ert_config.ert_templates,
        model_config=ert_config.model_config,
        runpaths=run_paths(ert_config),
    )
    assert (
        Path(run_arg[0].runpath) / "result.txt"
    ).read_text() == "I WANT TO REPLACE:my_custom_variable"


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "key, expected",
    [
        ("<DATE>", datetime.date(datetime.today()).isoformat()),
        ("<NUM_CPU>", "1"),
        ("<CONFIG_FILE_BASE>", "config"),
        ("<CONFIG_FILE>", "config.ert"),
        ("<ERT-CASE>", "prior"),
        ("<ERTCASE>", "prior"),
        ("<ECL_BASE>", "my_case0"),
        ("<ECLBASE>", "my_case0"),
        ("<IENS>", "0"),
        ("<ITER>", "0"),
    ],
)
def test_run_template_replace_in_file(
    key, expected, prior_ensemble, run_args, run_paths
):
    config_text = dedent(
        """
        NUM_REALIZATIONS 1
        JOBNAME my_case%d
        RUN_TEMPLATE template.tmpl result.txt
        """
    )
    Path("template.tmpl").write_text(f"I WANT TO REPLACE:{key}", encoding="utf-8")
    Path("config.ert").write_text(config_text, encoding="utf-8")

    ert_config = ErtConfig.from_file("config.ert")
    run_arg = run_args(ert_config, prior_ensemble)
    create_run_path(
        run_args=run_arg,
        ensemble=prior_ensemble,
        user_config_file=ert_config.user_config_file,
        env_vars=ert_config.env_vars,
        forward_model_steps=ert_config.forward_model_steps,
        substitutions=ert_config.substitutions,
        templates=ert_config.ert_templates,
        model_config=ert_config.model_config,
        runpaths=run_paths(ert_config),
    )
    assert (Path(run_arg[0].runpath) / "result.txt").read_text(
        encoding="utf-8"
    ) == f"I WANT TO REPLACE:{expected}"


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
def test_run_template_replace_in_ecl(
    ecl_base, expected_file, prior_ensemble, run_args, run_paths
):
    config_text = dedent(
        f"""
        NUM_REALIZATIONS 1
        ECLBASE {ecl_base}
        RUN_TEMPLATE BASE_ECL_FILE.DATA <ECLBASE>.DATA
        """
    )
    Path("BASE_ECL_FILE.DATA").write_text(
        "I WANT TO REPLACE:<NUM_CPU>", encoding="utf-8"
    )
    Path("config.ert").write_text(config_text, encoding="utf-8")

    ert_config = ErtConfig.from_file("config.ert")
    run_arg = run_args(ert_config, prior_ensemble)
    create_run_path(
        run_args=run_arg,
        ensemble=prior_ensemble,
        user_config_file=ert_config.user_config_file,
        env_vars=ert_config.env_vars,
        forward_model_steps=ert_config.forward_model_steps,
        substitutions=ert_config.substitutions,
        templates=ert_config.ert_templates,
        model_config=ert_config.model_config,
        runpaths=run_paths(ert_config),
    )
    assert (
        Path(run_arg[0].runpath) / expected_file
    ).read_text() == "I WANT TO REPLACE:1"


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "key, expected",
    [
        ("<DATE>", datetime.date(datetime.today()).isoformat()),
        ("<NUM_CPU>", "1"),
        ("<CONFIG_FILE_BASE>", "config"),
        ("<CONFIG_FILE>", "config.ert"),
        ("<ERT-CASE>", "prior"),
        ("<ERTCASE>", "prior"),
        ("<ECL_BASE>", "ECL_CASE0"),
        ("<ECLBASE>", "ECL_CASE0"),
        ("<IENS>", "0"),
        ("<ITER>", "0"),
    ],
)
def test_run_template_replace_in_ecl_data_file(
    key, expected, prior_ensemble, run_paths, run_args
):
    """
    This test that we copy the DATA_FILE into the runpath,
    do substitutions and rename it from the DATA_FILE name
    to ECLBASE
    """
    config_text = dedent(
        """
        NUM_REALIZATIONS 1
        ECLBASE ECL_CASE<IENS>
        DATA_FILE MY_DATA_FILE.DATA
        """
    )
    Path("MY_DATA_FILE.DATA").write_text(f"I WANT TO REPLACE:{key}", encoding="utf-8")
    Path("config.ert").write_text(config_text, encoding="utf-8")

    ert_config = ErtConfig.from_file("config.ert")
    run_arg = run_args(ert_config, prior_ensemble)
    create_run_path(
        run_args=run_arg,
        ensemble=prior_ensemble,
        user_config_file=ert_config.user_config_file,
        env_vars=ert_config.env_vars,
        forward_model_steps=ert_config.forward_model_steps,
        substitutions=ert_config.substitutions,
        templates=ert_config.ert_templates,
        model_config=ert_config.model_config,
        runpaths=run_paths(ert_config),
    )
    assert (Path(run_arg[0].runpath) / "ECL_CASE0.DATA").read_text(
        encoding="utf-8"
    ) == f"I WANT TO REPLACE:{expected}"


@pytest.mark.usefixtures("use_tmpdir")
def test_that_error_is_raised_when_data_file_is_badly_encoded(
    prior_ensemble, run_paths, run_args
):
    config_text = dedent(
        """
    NUM_REALIZATIONS 1
    ECLBASE ECL_CASE<IENS>
    DATA_FILE MY_DATA_FILE.DATA
    """
    )
    Path("MY_DATA_FILE.DATA").write_text("I WANT TO REPLACE:<DATE>", encoding="utf-8")
    Path("config.ert").write_text(config_text, encoding="utf-8")

    ert_config = ErtConfig.from_file("config.ert")

    Path("MY_DATA_FILE.DATA").write_text(
        "ä I WANT TO REPLACE:<DATE>", encoding="iso-8859-1"
    )

    with pytest.raises(
        ValueError,
        match="Unsupported non UTF-8 character found in file: .*MY_DATA_FILE.DATA",
    ):
        create_run_path(
            run_args=run_args(ert_config, prior_ensemble),
            ensemble=prior_ensemble,
            user_config_file=ert_config.user_config_file,
            env_vars=ert_config.env_vars,
            forward_model_steps=ert_config.forward_model_steps,
            substitutions=ert_config.substitutions,
            templates=ert_config.ert_templates,
            model_config=ert_config.model_config,
            runpaths=run_paths(ert_config),
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_run_template_replace_in_file_name(prior_ensemble, run_args, run_paths):
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
    Path("template.tmpl").write_text(
        "Not important, name of the file is important", encoding="utf-8"
    )
    Path("config.ert").write_text(config_text, encoding="utf-8")

    ert_config = ErtConfig.from_file("config.ert")
    run_arg = run_args(ert_config, prior_ensemble)
    create_run_path(
        run_args=run_arg,
        ensemble=prior_ensemble,
        user_config_file=ert_config.user_config_file,
        env_vars=ert_config.env_vars,
        forward_model_steps=ert_config.forward_model_steps,
        substitutions=ert_config.substitutions,
        templates=ert_config.ert_templates,
        model_config=ert_config.model_config,
        runpaths=run_paths(ert_config),
    )
    assert (
        Path(run_arg[0].runpath) / "result.txt"
    ).read_text() == "Not important, name of the file is important"


@pytest.mark.usefixtures("use_tmpdir")
def test_that_sampling_prior_makes_initialized_fs(storage):
    """
    This checks that creating the run path initializes the selected case,
    for that parameters are needed, so add a simple GEN_KW.
    """
    config_text = dedent(
        """
        NUM_REALIZATIONS 1
        JOBNAME my_case%d
        GEN_KW KW_NAME template.txt kw.txt prior.txt FORWARD_INIT:False
        """
    )
    Path("template.tmpl").write_text(
        "Not important, name of the file is important", encoding="utf-8"
    )
    Path("config.ert").write_text(config_text, encoding="utf-8")
    with open("template.txt", "w", encoding="utf-8") as fh:
        fh.writelines("MY_KEYWORD <MY_KEYWORD>")
    with open("prior.txt", "w", encoding="utf-8") as fh:
        fh.writelines("MY_KEYWORD NORMAL 0 1")

    ert_config = ErtConfig.from_file("config.ert")

    prior_ensemble = storage.create_ensemble(
        storage.create_experiment(
            parameters=ert_config.ensemble_config.parameter_configuration
        ),
        name="prior",
        ensemble_size=ert_config.model_config.num_realizations,
    )

    assert not prior_ensemble.is_initalized()
    sample_prior(prior_ensemble, [0])
    assert prior_ensemble.is_initalized()


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
            'upper' 'base' '*' 'data_file' 4 /

            -- Line above left intentionally blank
            'lower' 'base' '*' 'data_file_lower' /
            /"""
            ),
            6,
            id=(
                "Entry number 5 on each lines says how many cpus each "
                "slave should run on, omitting it means 1 cpu. "
                "1 for master, 4 for slave 1 and 1 for slave 2 = 6"
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
    Path("MY_DATA_FILE.DATA").write_text(eclipse_data, encoding="utf-8")
    Path("config.ert").write_text(config_text, encoding="utf-8")

    ert_config = ErtConfig.from_file("config.ert")
    assert int(ert_config.substitutions["<NUM_CPU>"]) == expected_cpus


@pytest.mark.filterwarnings(
    "ignore:.*RUNPATH keyword contains deprecated value placeholders.*:ert.config.ConfigWarning"
)
@pytest.mark.usefixtures("use_tmpdir")
def test_that_deprecated_runpath_substitution_remain_valid(
    prior_ensemble, run_paths, run_args
):
    """
    This checks that deprecated runpath substitution, using %d, remain intact.
    """
    config_text = dedent(
        """
        NUM_REALIZATIONS 2
        JOBNAME my_case%d
        RUNPATH realization-%d/iter-%d
        FORWARD_MODEL COPY_DIRECTORY(<FROM>=<CONFIG_PATH>/, <TO>=<RUNPATH>/)
        """
    )
    Path("config.ert").write_text(config_text, encoding="utf-8")

    ert_config = ErtConfig.with_plugins().from_file("config.ert")

    run_arg = run_args(ert_config, prior_ensemble, 2)
    create_run_path(
        run_args=run_arg,
        ensemble=prior_ensemble,
        user_config_file=ert_config.user_config_file,
        env_vars=ert_config.env_vars,
        forward_model_steps=ert_config.forward_model_steps,
        substitutions=ert_config.substitutions,
        templates=ert_config.ert_templates,
        model_config=ert_config.model_config,
        runpaths=run_paths(ert_config),
    )

    for realization in run_arg:
        assert str(Path().absolute()) + "/realization-" + str(
            realization.iens
        ) + "/iter-0" in Path(realization.runpath + "/jobs.json").read_text(
            encoding="utf-8"
        )


@pytest.mark.parametrize("itr", [0, 1, 2, 17])
def test_write_snakeoil_runpath_file(snake_oil_case, storage, itr):
    ert_config = snake_oil_case
    experiment_id = storage.create_experiment(
        parameters=ert_config.ensemble_config.parameter_configuration
    )
    prior_ensemble = storage.create_ensemble(
        experiment_id, name="prior", ensemble_size=25, iteration=itr
    )

    num_realizations = 25
    mask = [True] * num_realizations
    mask[13] = False
    runpath_fmt = (
        "simulations/<GEO_ID>/realization-<IENS>/iter-<ITER>/"
        "magic-real-<IENS>/magic-iter-<ITER>"
    )
    jobname_fmt = "SNAKE_OIL_%d"
    global_substitutions = ert_config.substitutions
    for i in range(num_realizations):
        global_substitutions[f"<GEO_ID_{i}_{itr}>"] = str(10 * i)
    run_paths = Runpaths(
        jobname_format=jobname_fmt,
        runpath_format=runpath_fmt,
        filename=str("a_file_name"),
        substitutions=global_substitutions,
    )
    sample_prior(prior_ensemble, [i for i, active in enumerate(mask) if active])
    run_args = create_run_arguments(
        run_paths,
        [True, True],
        prior_ensemble,
    )
    create_run_path(
        run_args=run_args,
        ensemble=prior_ensemble,
        user_config_file=ert_config.user_config_file,
        env_vars=ert_config.env_vars,
        forward_model_steps=ert_config.forward_model_steps,
        substitutions=ert_config.substitutions,
        templates=ert_config.ert_templates,
        model_config=ert_config.model_config,
        runpaths=run_paths,
    )

    for run_arg in run_args:
        if not run_arg.active:
            continue
        assert os.path.isdir(f"simulations/{10*run_arg.iens}")

    runpath_list_path = "a_file_name"
    assert os.path.isfile(runpath_list_path)

    exp_runpaths = [
        runpath_fmt.replace("<ITER>", str(itr))
        .replace("<IENS>", str(run_arg.iens))
        .replace("<GEO_ID>", str(10 * run_arg.iens))
        for run_arg in run_args
        if run_arg.active
    ]
    exp_runpaths = list(map(os.path.realpath, exp_runpaths))

    with open(runpath_list_path, "r", encoding="utf-8") as f:
        dumped_runpaths = list(zip(*[line.split() for line in f.readlines()]))[1]

    assert list(exp_runpaths) == list(dumped_runpaths)


@pytest.mark.usefixtures("use_tmpdir")
def test_assert_export(prior_ensemble, run_args, run_paths):
    # Write a minimal config file with env
    with open("config_file.ert", "w", encoding="utf-8") as fout:
        fout.write(
            dedent(
                """
        NUM_REALIZATIONS 1
        JOBNAME a_name_%d
        RUNPATH_FILE directory/test_runpath_list.txt
        """
            )
        )
    ert_config = ErtConfig.from_file("config_file.ert")
    runpath_list_file = ert_config.runpath_file
    assert not runpath_list_file.exists()

    sample_prior(prior_ensemble, [0])
    run_arg = run_args(ert_config, prior_ensemble)
    create_run_path(
        run_args=run_arg,
        ensemble=prior_ensemble,
        user_config_file=ert_config.user_config_file,
        env_vars=ert_config.env_vars,
        forward_model_steps=ert_config.forward_model_steps,
        substitutions=ert_config.substitutions,
        templates=ert_config.ert_templates,
        model_config=ert_config.model_config,
        runpaths=run_paths(ert_config),
    )
    assert runpath_list_file.exists()
    assert runpath_list_file.name == "test_runpath_list.txt"
    assert (
        runpath_list_file.read_text("utf-8")
        == f"000  {os.getcwd()}/simulations/realization-0/iter-0  a_name_0  000\n"
    )


def _create_runpath(ert_config: ErtConfig, storage: Storage) -> None:
    """
    Instantiate an ERT runpath. This will create the parameter coefficients.
    """
    ensemble = storage.create_ensemble(
        storage.create_experiment(),
        name="prior",
        ensemble_size=ert_config.model_config.num_realizations,
    )
    run_paths = Runpaths(
        jobname_format=ert_config.model_config.jobname_format_string,
        runpath_format=ert_config.model_config.runpath_format_string,
        filename=str(ert_config.runpath_file),
        substitutions=ert_config.substitutions,
    )
    create_run_path(
        run_args=create_run_arguments(
            run_paths, [True] * ensemble.ensemble_size, ensemble
        ),
        ensemble=ensemble,
        user_config_file=ert_config.user_config_file,
        env_vars=ert_config.env_vars,
        forward_model_steps=ert_config.forward_model_steps,
        substitutions=ert_config.substitutions,
        templates=ert_config.ert_templates,
        model_config=ert_config.model_config,
        runpaths=run_paths,
    )


@pytest.mark.parametrize(
    "append,numcpu",
    [
        ("", 1),  # Default is 1
        ("NUM_CPU 2\n", 2),
        ("DATA_FILE DATA\n", 8),  # Data file dictates NUM_CPU with PARALLEL
        ("NUM_CPU 3\nDATA_FILE DATA\n", 3),  # Explicit NUM_CPU supersedes PARALLEL
    ],
)
def test_num_cpu_subst(
    monkeypatch, tmp_path, append, numcpu, storage, run_paths, run_args
):
    """
    Make sure that <NUM_CPU> is substituted to the correct values
    """
    monkeypatch.chdir(tmp_path)

    (tmp_path / "test.ert").write_text(
        "JOBNAME test_%d\n"
        "NUM_REALIZATIONS 1\n"
        "INSTALL_JOB dump DUMP\n"
        "FORWARD_MODEL dump\n" + append
    )
    (tmp_path / "DATA").write_text("PARALLEL 8 /")
    (tmp_path / "DUMP").write_text("EXECUTABLE echo\nARGLIST <NUM_CPU>\n")

    config = ErtConfig.from_file(str(tmp_path / "test.ert"))
    _create_runpath(config, storage)

    with open("simulations/realization-0/iter-0/jobs.json", encoding="utf-8") as f:
        jobs = orjson.loads(f.read())
        assert [str(numcpu)] == jobs["jobList"][0]["argList"]


@pytest.mark.parametrize(
    "run_path, expected_raise, msg",
    [
        ("simulations/realization-<IENS>/iter-<ITER>", False, ""),
        ("simulations/realization-%d/iter-%d", False, ""),
        ("simulations/realization-%d", False, ""),
        (
            "simulations/realization-<IENS>/iter-%d",
            True,
            "RUNPATH cannot combine deprecated ",
        ),
        (
            "simulations/realization-<IENS>/iter-<IENS>",
            True,
            "RUNPATH cannot contain multiple <IENS>",
        ),
        (
            "simulations/realization-<ITER>/iter-<ITER>",
            True,
            "RUNPATH cannot contain multiple <ITER>",
        ),
        (
            "simulations/realization-%d/iter-%d/more-%d",
            True,
            "RUNPATH cannot contain more than two",
        ),
    ],
)
@pytest.mark.filterwarnings(
    "ignore:.*RUNPATH keyword contains deprecated value placeholders.*:ert.config.ConfigWarning"
)
@pytest.mark.usefixtures("use_tmpdir")
def test_that_runpaths_are_raised_when_invalid(run_path, expected_raise, msg):
    """
    This checks that RUNPATH does not include too many or few substitution placeholders
    """
    config_text = dedent(
        """
        NUM_REALIZATIONS 2
        """
    )
    Path("config.ert").write_text(config_text + f"RUNPATH {run_path}", encoding="utf-8")

    if expected_raise:
        with pytest.raises(
            ConfigValidationError,
            match=f"{msg}.*{run_path}`.",
        ):
            _ = ErtConfig.from_file("config.ert")
    else:
        _ = ErtConfig.from_file("config.ert")


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "placeholder",
    ["<ERTCASE>", "<ERT-CASE>"],
)
def test_assert_ertcase_replaced_in_runpath(placeholder, prior_ensemble, storage):
    # Write a minimal config file with env
    with open("config_file.ert", "w", encoding="utf-8") as fout:
        fout.write(
            dedent(
                f"""
        NUM_REALIZATIONS 1
        JOBNAME a_name_%d
        RUNPATH simulations/{placeholder}/realization-<IENS>/iter-<ITER>
        """
            )
        )
    ert_config = ErtConfig.from_file("config_file.ert")
    _create_runpath(ert_config, storage)

    runpath_file = (
        f"{os.getcwd()}/simulations/{prior_ensemble.name}/realization-0/iter-0"
    )

    assert (
        ert_config.runpath_file.read_text("utf-8")
        == f"000  {runpath_file}  a_name_0  000\n"
    )
    assert Path(runpath_file).exists()
    jobs_json = Path(runpath_file) / "jobs.json"
    assert jobs_json.exists()


@pytest.mark.parametrize("itr", [0, 1, 2, 17])
def test_crete_runpath_adds_manifest_to_runpath(snake_oil_case, storage, itr):
    ert_config = snake_oil_case
    experiment_id = storage.create_experiment(
        parameters=ert_config.ensemble_config.parameter_configuration,
        responses=ert_config.ensemble_config.response_configuration,
    )
    prior_ensemble = storage.create_ensemble(
        experiment_id, name="prior", ensemble_size=25, iteration=itr
    )

    num_realizations = 25
    runpath_fmt = (
        "simulations/<GEO_ID>/realization-<IENS>/iter-<ITER>/"
        "magic-real-<IENS>/magic-iter-<ITER>"
    )
    global_substitutions = ert_config.substitutions
    for i in range(num_realizations):
        global_substitutions[f"<GEO_ID_{i}_{itr}>"] = str(10 * i)

    run_paths = Runpaths(
        jobname_format="SNAKE_OIL_%d",
        runpath_format=runpath_fmt,
        filename="a_file_name",
        substitutions=global_substitutions,
    )

    sample_prior(prior_ensemble, range(num_realizations))
    run_args = create_run_arguments(
        run_paths,
        [True, True],
        prior_ensemble,
    )

    create_run_path(
        run_args=run_args,
        ensemble=prior_ensemble,
        user_config_file=ert_config.user_config_file,
        env_vars=ert_config.env_vars,
        forward_model_steps=ert_config.forward_model_steps,
        substitutions=ert_config.substitutions,
        templates=ert_config.ert_templates,
        model_config=ert_config.model_config,
        runpaths=run_paths,
    )

    exp_runpaths = [
        runpath_fmt.replace("<ITER>", str(itr))
        .replace("<IENS>", str(run_arg.iens))
        .replace("<GEO_ID>", str(10 * run_arg.iens))
        for run_arg in run_args
        if run_arg.active
    ]
    exp_runpaths = list(map(os.path.realpath, exp_runpaths))
    expected_manifest_values = {
        "snake_oil_params.txt",
        "snake_oil_opr_diff_199.txt",
        "snake_oil_wpr_diff_199.txt",
        "snake_oil_gpr_diff_199.txt",
        "SNAKE_OIL_FIELD.UNSMRY",
        "SNAKE_OIL_FIELD.SMSPEC",
    }
    for run_path in exp_runpaths:
        manifest_path = Path(run_path) / "manifest.json"
        assert manifest_path.exists()
        with open(manifest_path, encoding="utf-8") as f:
            manifest = orjson.loads(f.read())
            assert set(manifest.values()) == expected_manifest_values
