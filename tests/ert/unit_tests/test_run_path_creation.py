import asyncio
import os
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import numpy as np
import orjson
import pytest
import xtgeo

from ert.callbacks import forward_model_ok
from ert.config import (
    ConfigValidationError,
    ErtConfig,
    Field,
    GenKwConfig,
    SurfaceConfig,
)
from ert.enkf_main import create_run_path, sample_prior
from ert.load_status import LoadStatus
from ert.run_arg import create_run_arguments
from ert.runpaths import Runpaths
from ert.storage import Storage
from tests.ert.unit_tests.config.egrid_generator import simple_grid
from tests.ert.unit_tests.config.summary_generator import simple_smspec, simple_unsmry


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
        run_arg,
        prior_ensemble,
        ert_config,
        run_paths(ert_config),
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
        run_arg,
        prior_ensemble,
        ert_config,
        run_paths(ert_config),
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
        run_arg,
        prior_ensemble,
        ert_config,
        run_paths(ert_config),
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
        run_arg,
        prior_ensemble,
        ert_config,
        run_paths(ert_config),
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
        run_arg,
        prior_ensemble,
        ert_config,
        run_paths(ert_config),
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
            run_args(ert_config, prior_ensemble),
            prior_ensemble,
            ert_config,
            run_paths(ert_config),
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
        run_arg,
        prior_ensemble,
        ert_config,
        run_paths(ert_config),
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
        run_arg,
        prior_ensemble,
        ert_config,
        run_paths(ert_config),
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
        run_args,
        prior_ensemble,
        ert_config,
        run_paths,
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
        run_arg,
        prior_ensemble,
        ert_config,
        run_paths(ert_config),
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
        create_run_arguments(run_paths, [True] * ensemble.ensemble_size, ensemble),
        ensemble,
        ert_config,
        run_paths,
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


def save_zeros(prior_ensemble, num_realizations, dim_size):
    parameter_configs = prior_ensemble.experiment.parameter_configuration
    for parameter, config_node in parameter_configs.items():
        for realization_nr in range(num_realizations):
            if isinstance(config_node, SurfaceConfig):
                config_node.save_parameters(
                    prior_ensemble, parameter, realization_nr, np.zeros(dim_size**2)
                )
            elif isinstance(config_node, Field):
                config_node.save_parameters(
                    prior_ensemble, parameter, realization_nr, np.zeros(dim_size**3)
                )
            elif isinstance(config_node, GenKwConfig):
                config_node.save_parameters(
                    prior_ensemble, parameter, realization_nr, np.zeros(1)
                )
            else:
                raise ValueError(f"unexpected {config_node}")


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize("itr", [0, 1])
def test_when_manifest_files_are_written_forward_model_ok_succeeds(storage, itr):
    num_realizations = 2
    dim_size = 2
    simple_grid().to_file("GRID.EGRID")
    Path("base.irap").write_text("", encoding="utf-8")
    xtgeo.RegularSurface(ncol=dim_size, nrow=dim_size, xinc=1, yinc=1).to_file(
        "base.irap", fformat="irap_ascii"
    )
    for i in range(num_realizations):
        xtgeo.RegularSurface(ncol=dim_size, nrow=dim_size, xinc=1, yinc=1).to_file(
            f"{i}init{i}0.irap", fformat="irap_ascii"
        )
        xtgeo.GridProperty(
            ncol=2,
            nrow=2,
            nlay=2,
            name="PORO1",
            values=np.zeros((dim_size, dim_size, dim_size)),
        ).to_file(f"{i}init{i}0.roff", fformat="roff")
        Path(f"{i}gen_init{i}0.txt").write_text("PARMA 1.0\n", encoding="utf-8")
    Path("gen0.txt").write_text("PARMA NORMAL 0 1\n", encoding="utf-8")
    Path("gen1.txt").write_text("PARMA NORMAL 0 1\n", encoding="utf-8")
    Path("template.txt").write_text("<PARMA>", encoding="utf-8")

    config = ErtConfig.from_file_contents(
        dedent(
            f"""\
            NUM_REALIZATIONS {num_realizations}
            DEFINE <ALL> -<ITER>-<IENS>-<GEO_ID>
            DEFINE <GEO_ID_0_0> HELLO0
            DEFINE <GEO_ID_1_0> HELLO1
            DEFINE <GEO_ID_0_1> HELLO0
            DEFINE <GEO_ID_1_1> HELLO1

            RUNPATH simulations/<GEO_ID>/realization-<IENS>/iter-<ITER>

            ECLBASE CASE<ALL>
            GRID GRID.EGRID
            SUMMARY FOPR

            GEN_DATA GENDATA RESULT_FILE:gen_data<ALL>.txt

            SURFACE SURF1 OUTPUT_FILE:surf1_output<ALL>.irap BASE_SURFACE:base.irap FORWARD_INIT:True INIT_FILES:surf1_init<ALL>.irap
            SURFACE SURF2 OUTPUT_FILE:surf2_output<ALL>.irap BASE_SURFACE:base.irap INIT_FILES:%dinit<IENS><ITER>.irap

            FIELD PORO0 PARAMETER field1<ALL>.roff INIT_FILES:field1_init<ALL>.roff FORWARD_INIT:TRUE
            FIELD PORO1 PARAMETER field2<ALL>.roff INIT_FILES:%dinit<IENS><ITER>.roff

            GEN_KW GEN0 gen0.txt INIT_FILES:%dgen_init<IENS><ITER>.txt
            GEN_KW GEN1 template.txt gen_parameter.txt gen1.txt INIT_FILES:%dgen_init<IENS><ITER>.txt
            """
        )
    )

    experiment_id = storage.create_experiment(
        parameters=config.ensemble_config.parameter_configuration,
        responses=config.ensemble_config.response_configuration,
    )
    prior_ensemble = storage.create_ensemble(
        experiment_id, name="prior", ensemble_size=num_realizations, iteration=itr
    )

    run_paths = Runpaths(
        jobname_format=config.model_config.jobname_format_string,
        runpath_format=config.model_config.runpath_format_string,
        filename=str(config.runpath_file),
        substitutions=config.substitutions,
        eclbase=config.model_config.eclbase_format_string,
    )

    if itr == 0:
        sample_prior(prior_ensemble, range(num_realizations))
    else:
        save_zeros(prior_ensemble, num_realizations, dim_size=dim_size)

    run_args = create_run_arguments(
        run_paths,
        [True, True],
        prior_ensemble,
    )

    create_run_path(run_args, prior_ensemble, config, run_paths)

    for i, run_path in enumerate(run_paths.get_paths(range(num_realizations), itr)):
        manifest_path = Path(run_path) / "manifest.json"
        assert manifest_path.exists()
        expected_files = {
            run_path + f"/CASE-{itr}-{i}-HELLO{i}.UNSMRY",
            run_path + f"/CASE-{itr}-{i}-HELLO{i}.SMSPEC",
            run_path + f"/gen_data-{itr}-{i}-HELLO{i}.txt",
        }.union(
            {
                run_path + f"/field1_init-{itr}-{i}-HELLO{i}.roff",
                run_path + f"/surf1_init-{itr}-{i}-HELLO{i}.irap",
            }
            if itr == 0
            else set()
        )
        with open(manifest_path, encoding="utf-8") as f:
            manifest = orjson.loads(f.read())
            assert set(manifest.values()) == expected_files

        # write files in manifest
        for file in expected_files:
            if file.endswith("roff"):
                xtgeo.GridProperty(
                    ncol=2,
                    nrow=2,
                    nlay=2,
                    name="PORO0",
                    values=np.zeros((2, 2, 2)),
                ).to_file(file, fformat="roff")
            elif file.endswith("irap"):
                xtgeo.RegularSurface(ncol=2, nrow=3, xinc=1, yinc=1).to_file(
                    file, fformat="irap_ascii"
                )
            elif file.endswith("UNSMRY"):
                simple_unsmry().to_file(file)
            elif file.endswith("SMSPEC"):
                simple_smspec().to_file(file)
            elif file.endswith("txt"):
                Path(file).write_text("1.0", encoding="utf-8")
            else:
                raise AssertionError()

    # When files in manifest are written we expect forward_model_ok to succeed
    for run_arg in run_args:
        load_result = asyncio.run(forward_model_ok(run_arg))
        assert load_result.status == LoadStatus.LOAD_SUCCESSFUL
