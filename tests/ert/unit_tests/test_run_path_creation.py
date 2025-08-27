import asyncio
import os
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import numpy as np
import orjson
import pytest
import xtgeo

from ert.config import (
    ConfigValidationError,
    ErtConfig,
    Field,
    GenKwConfig,
    SurfaceConfig,
)
from ert.enkf_main import create_run_path, sample_prior
from ert.run_arg import create_run_arguments
from ert.runpaths import Runpaths
from ert.storage import (
    RealizationStorageState,
    load_realization_parameters_and_responses,
)
from tests.ert.unit_tests.config.egrid_generator import simple_grid
from tests.ert.unit_tests.config.summary_generator import simple_smspec, simple_unsmry

config_contents = """\
NUM_REALIZATIONS 1
QUEUE_SYSTEM LOCAL
ENSPATH storage
{parameters}
"""


@pytest.fixture
def make_run_path(run_paths, run_args, storage):
    def func(ert_config):
        experiment_id = storage.create_experiment(
            parameters=ert_config.ensemble_config.parameter_configuration,
            templates=ert_config.ert_templates,
        )
        prior_ensemble = storage.create_ensemble(
            experiment_id, name="prior", ensemble_size=1
        )
        sample_prior(prior_ensemble, [0], 123)
        runargs = run_args(ert_config, prior_ensemble, 1)
        runpaths = run_paths(ert_config)
        create_run_path(
            run_args=runargs,
            ensemble=prior_ensemble,
            user_config_file=ert_config.user_config_file,
            forward_model_steps=ert_config.forward_model_steps,
            env_vars=ert_config.env_vars,
            env_pr_fm_step=ert_config.env_pr_fm_step,
            substitutions=ert_config.substitutions,
            parameters_file="parameters",
            runpaths=runpaths,
        )
        return prior_ensemble, runargs, runpaths

    return func


@pytest.mark.usefixtures("use_tmpdir")
def test_setup_with_gen_kw_generates_parameters_txt(make_run_path):
    Path("genkw").write_text("genkw0 UNIFORM 0 1", encoding="utf-8")
    ert_config = ErtConfig.from_file_contents(
        config_contents.format(parameters="GEN_KW GENKW genkw")
    )
    make_run_path(ert_config)
    assert os.path.exists("simulations/realization-0/iter-0")
    assert os.path.exists("simulations/realization-0/iter-0/parameters.txt")
    assert len(os.listdir("simulations")) == 1
    assert len(os.listdir("simulations/realization-0")) == 1


@pytest.mark.usefixtures("use_tmpdir")
def test_setup_without_gen_kw_does_not_generates_parameters_txt(make_run_path):
    ert_config = ErtConfig.from_file_contents(config_contents.format(parameters=""))
    make_run_path(ert_config)
    assert os.path.exists("simulations/realization-0/iter-0")
    assert not os.path.exists("simulations/realization-0/iter-0/parameters.txt")
    assert len(os.listdir("simulations")) == 1
    assert len(os.listdir("simulations/realization-0")) == 1


@pytest.mark.usefixtures("use_tmpdir")
def test_jobs_json_is_backed_up(make_run_path):
    Path("genkw").write_text("genkw0 UNIFORM 0 1", encoding="utf-8")
    ert_config = ErtConfig.from_file_contents(
        config_contents.format(parameters="GEN_KW GENKW genkw")
    )
    make_run_path(ert_config)
    assert os.path.exists("simulations/realization-0/iter-0/jobs.json")
    make_run_path(ert_config)
    iter0_output_files = os.listdir("simulations/realization-0/iter-0/")
    assert len([f for f in iter0_output_files if f.startswith("jobs.json")]) > 1, (
        "No backup created for jobs.json"
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_run_template_replace_symlink_does_not_write_to_source(
    prior_ensemble_args, run_args, run_paths
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
    Path("template.tmpl").write_text("I want to replace: <IENS>", encoding="utf-8")
    ert_config = ErtConfig.from_file_contents(
        dedent(
            """\
            NUM_REALIZATIONS 1
            RUN_TEMPLATE template.tmpl result.txt
            """
        )
    )
    prior_ensemble = prior_ensemble_args(templates=ert_config.ert_templates)
    run_arg = run_args(ert_config, prior_ensemble)
    run_path = Path(run_arg[0].runpath)
    os.makedirs(run_path)
    # Write a file that will be symlinked into the run run path with the
    # same name as the target_file
    Path("start.txt").write_text(
        "I don't want to replace in this file", encoding="utf-8"
    )
    os.symlink("start.txt", run_path / "result.txt")
    create_run_path(
        run_args=run_arg,
        ensemble=prior_ensemble,
        user_config_file=ert_config.user_config_file,
        env_vars=ert_config.env_vars,
        env_pr_fm_step=ert_config.env_pr_fm_step,
        forward_model_steps=ert_config.forward_model_steps,
        substitutions=ert_config.substitutions,
        parameters_file="parameters",
        runpaths=run_paths(ert_config),
    )
    assert (run_path / "result.txt").read_text(
        encoding="utf-8"
    ) == "I want to replace: 0"
    # Check that the source of the symlinked file is not updated
    assert (
        Path("start.txt").read_text(encoding="utf-8")
        == "I don't want to replace in this file"
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_run_template_replace_in_file_with_custom_define(make_run_path):
    """
    This test checks that we are able to magically replace custom magic
    strings using the DEFINE keyword
    """
    Path("template.tmpl").write_text("I WANT TO REPLACE:<MY_VAR>", encoding="utf-8")
    ert_config = ErtConfig.from_file_contents(
        dedent(
            """\
            NUM_REALIZATIONS 1
            DEFINE <MY_VAR> my_custom_variable
            RUN_TEMPLATE template.tmpl result.txt
            """
        )
    )
    _, run_arg, _ = make_run_path(ert_config)

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
def test_run_template_replace_in_file(key, expected, make_run_path):
    Path("template.tmpl").write_text(f"I WANT TO REPLACE:{key}", encoding="utf-8")
    ert_config = ErtConfig.from_file_contents(
        dedent(
            """\
            NUM_REALIZATIONS 1
            JOBNAME my_case%d
            RUN_TEMPLATE template.tmpl result.txt
            """
        )
    )
    _, run_arg, _ = make_run_path(ert_config)

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
def test_run_template_replace_in_ecl(ecl_base, expected_file, make_run_path):
    Path("BASE_ECL_FILE.DATA").write_text(
        "I WANT TO REPLACE:<NUM_CPU>", encoding="utf-8"
    )
    ert_config = ErtConfig.from_file_contents(
        dedent(
            f"""\
            NUM_REALIZATIONS 1
            ECLBASE {ecl_base}
            RUN_TEMPLATE BASE_ECL_FILE.DATA <ECLBASE>.DATA
            """
        )
    )
    _, run_arg, _ = make_run_path(ert_config)
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
def test_run_template_replace_in_ecl_data_file(key, expected, make_run_path):
    """
    This test that we copy the DATA_FILE into the runpath,
    do substitutions and rename it from the DATA_FILE name
    to ECLBASE
    """
    Path("MY_DATA_FILE.DATA").write_text(f"I WANT TO REPLACE:{key}", encoding="utf-8")
    ert_config = ErtConfig.from_file_contents(
        dedent(
            """\
        NUM_REALIZATIONS 1
        ECLBASE ECL_CASE<IENS>
        DATA_FILE MY_DATA_FILE.DATA
        """
        )
    )
    _, run_arg, _ = make_run_path(ert_config)
    assert (Path(run_arg[0].runpath) / "ECL_CASE0.DATA").read_text(
        encoding="utf-8"
    ) == f"I WANT TO REPLACE:{expected}"


@pytest.mark.usefixtures("use_tmpdir")
def test_that_error_is_raised_when_data_file_is_badly_encoded(make_run_path):
    Path("MY_DATA_FILE.DATA").write_text("I WANT TO REPLACE:<DATE>", encoding="utf-8")

    ert_config = ErtConfig.from_file_contents(
        dedent(
            """\
            NUM_REALIZATIONS 1
            ECLBASE ECL_CASE<IENS>
            DATA_FILE MY_DATA_FILE.DATA
            """
        )
    )

    Path("MY_DATA_FILE.DATA").write_text(
        "ä I WANT TO REPLACE:<DATE>", encoding="iso-8859-1"
    )
    err_str = (
        "Unsupported non UTF-8 character found in file: templates/MY_DATA_FILE_0.DATA"
    )
    with pytest.raises(
        ValueError,
        match=err_str,
    ):
        make_run_path(ert_config)


@pytest.mark.usefixtures("use_tmpdir")
def test_run_template_replace_in_file_name(make_run_path):
    """
    This test checks that we are able to magically replace custom magic
    strings using the DEFINE keyword
    """
    Path("template.tmpl").write_text(
        "Not important, name of the file is important", encoding="utf-8"
    )
    ert_config = ErtConfig.from_file_contents(
        dedent(
            """\
        NUM_REALIZATIONS 1
        DEFINE <MY_FILE_NAME> result.txt
        RUN_TEMPLATE template.tmpl <MY_FILE_NAME>
        """
        )
    )
    _, run_arg, _ = make_run_path(ert_config)
    assert (
        Path(run_arg[0].runpath) / "result.txt"
    ).read_text() == "Not important, name of the file is important"


@pytest.mark.usefixtures("use_tmpdir")
def test_that_sampling_prior_makes_initialized_fs(storage):
    """
    This checks that creating the run path initializes the selected case,
    for that parameters are needed, so add a simple GEN_KW.
    """
    Path("template.tmpl").write_text("Unimportant", encoding="utf-8")
    Path("template.txt").write_text("MY_KEYWORD <MY_KEYWORD>", encoding="utf-8")
    Path("prior.txt").write_text("MY_KEYWORD NORMAL 0 1", encoding="utf-8")

    ert_config = ErtConfig.from_file_contents(
        dedent(
            """\
            NUM_REALIZATIONS 2
            GEN_KW KW_NAME template.txt kw.txt prior.txt FORWARD_INIT:False
            """
        )
    )

    prior_ensemble = storage.create_ensemble(
        storage.create_experiment(
            parameters=ert_config.ensemble_config.parameter_configuration
        ),
        name="prior",
        ensemble_size=ert_config.runpath_config.num_realizations,
    )

    assert (
        RealizationStorageState.PARAMETERS_LOADED
        not in prior_ensemble.get_ensemble_state()[0]
    )
    assert (
        RealizationStorageState.PARAMETERS_LOADED
        not in prior_ensemble.get_ensemble_state()[1]
    )
    sample_prior(prior_ensemble, [0], 123)
    # Realization 0 state now contains PARAMETERS_LOADED
    assert (
        RealizationStorageState.PARAMETERS_LOADED
        in prior_ensemble.get_ensemble_state()[0]
    )
    assert (
        RealizationStorageState.PARAMETERS_LOADED
        not in prior_ensemble.get_ensemble_state()[1]
    )


@pytest.mark.parametrize(
    "eclipse_data, expected_cpus",
    [
        ("PARALLEL 4 /", 4),
        pytest.param(
            dedent(
                """\
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
    Path("MY_DATA_FILE.DATA").write_text(eclipse_data, encoding="utf-8")

    ert_config = ErtConfig.from_file_contents(
        dedent(
            """\
            NUM_REALIZATIONS 1
            DATA_FILE MY_DATA_FILE.DATA
            """
        )
    )
    assert int(ert_config.substitutions["<NUM_CPU>"]) == expected_cpus


@pytest.mark.filterwarnings(
    "ignore:.*RUNPATH keyword contains deprecated value "
    "placeholders.*:ert.config.ConfigWarning"
)
@pytest.mark.usefixtures("use_tmpdir")
def test_that_deprecated_runpath_substitution_remain_valid(make_run_path):
    """
    This checks that deprecated runpath substitution, using %d, remain intact.
    """
    ert_config = ErtConfig.with_plugins().from_file_contents(
        dedent(
            """\
            NUM_REALIZATIONS 2
            RUNPATH realization-%d/iter-%d
            FORWARD_MODEL COPY_DIRECTORY(<FROM>=<CONFIG_PATH>/, <TO>=<RUNPATH>/)
            """
        )
    )

    _, run_arg, _ = make_run_path(ert_config)

    for realization in run_arg:
        assert str(Path().absolute()) + "/realization-" + str(
            realization.iens
        ) + "/iter-0" in Path(realization.runpath + "/jobs.json").read_text(
            encoding="utf-8"
        )


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize("itr", [0, 1, 2, 17])
def test_write_runpath_file(storage, itr, run_paths):
    runpath_fmt = "simulations/<GEO_ID>/realization-<IENS>/iter-<ITER>"
    runpath_list_path = "a_file_name"
    ert_config = ErtConfig.from_file_contents(
        dedent(
            f"""\
            NUM_REALIZATIONS 25
            RUNPATH {runpath_fmt}
            RUNPATH_FILE {runpath_list_path}
            """
        )
    )
    num_realizations = ert_config.runpath_config.num_realizations
    experiment_id = storage.create_experiment(
        parameters=ert_config.ensemble_config.parameter_configuration
    )
    prior_ensemble = storage.create_ensemble(
        experiment_id, name="prior", ensemble_size=num_realizations, iteration=itr
    )

    mask = [True] * num_realizations
    mask[13] = False
    global_substitutions = ert_config.substitutions
    for i in range(num_realizations):
        global_substitutions[f"<GEO_ID_{i}_{itr}>"] = str(10 * i)
    run_path = run_paths(ert_config)
    sample_prior(prior_ensemble, [i for i, active in enumerate(mask) if active], 123)
    run_args = create_run_arguments(
        run_path,
        [True, True],
        prior_ensemble,
    )
    create_run_path(
        run_args=run_args,
        ensemble=prior_ensemble,
        user_config_file=ert_config.user_config_file,
        env_vars=ert_config.env_vars,
        env_pr_fm_step=ert_config.env_pr_fm_step,
        forward_model_steps=ert_config.forward_model_steps,
        substitutions=ert_config.substitutions,
        parameters_file="parameters",
        runpaths=run_path,
    )

    for run_arg in run_args:
        if not run_arg.active:
            continue
        assert os.path.isdir(f"simulations/{10 * run_arg.iens}")

    assert os.path.isfile(runpath_list_path)

    exp_runpaths = [
        runpath_fmt.replace("<ITER>", str(itr))
        .replace("<IENS>", str(run_arg.iens))
        .replace("<GEO_ID>", str(10 * run_arg.iens))
        for run_arg in run_args
        if run_arg.active
    ]
    exp_runpaths = list(map(os.path.realpath, exp_runpaths))

    with open(runpath_list_path, encoding="utf-8") as f:
        dumped_runpaths = list(zip(*[line.split() for line in f], strict=False))[1]

    assert list(exp_runpaths) == list(dumped_runpaths)


@pytest.mark.usefixtures("use_tmpdir")
def test_assert_export(make_run_path):
    ert_config = ErtConfig.from_file_contents(
        dedent(
            """\
            NUM_REALIZATIONS 1
            JOBNAME a_name_%d
            RUNPATH_FILE directory/test_runpath_list.txt
            """
        )
    )
    runpath_list_file = ert_config.runpath_file
    assert not runpath_list_file.exists()

    make_run_path(ert_config)

    assert runpath_list_file.exists()
    assert runpath_list_file.name == "test_runpath_list.txt"
    assert (
        runpath_list_file.read_text("utf-8")
        == f"000  {os.getcwd()}/simulations/realization-0/iter-0  a_name_0  000\n"
    )


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "append,numcpu",
    [
        ("", 1),  # Default is 1
        ("NUM_CPU 2\n", 2),
        ("DATA_FILE DATA\n", 8),  # Data file dictates NUM_CPU with PARALLEL
        ("NUM_CPU 3\nDATA_FILE DATA\n", 3),  # Explicit NUM_CPU supersedes PARALLEL
    ],
)
def test_num_cpu_subst(append, numcpu, make_run_path):
    """
    Make sure that <NUM_CPU> is substituted to the correct values
    """
    Path("DATA").write_text("PARALLEL 8 /", encoding="utf-8")
    Path("DUMP").write_text("EXECUTABLE echo\nARGLIST <NUM_CPU>\n", encoding="utf-8")

    config = ErtConfig.from_file_contents(
        "NUM_REALIZATIONS 1\nINSTALL_JOB dump DUMP\nFORWARD_MODEL dump\n" + append
    )
    make_run_path(config)

    with open("simulations/realization-0/iter-0/jobs.json", encoding="utf-8") as f:
        jobs = orjson.loads(f.read())
        assert [str(numcpu)] == jobs["jobList"][0]["argList"]


@pytest.mark.filterwarnings(
    "ignore:.*RUNPATH keyword contains deprecated "
    "value placeholders.*:ert.config.ConfigWarning"
)
@pytest.mark.parametrize(
    "iens_placeholder, iter_placeholder", [("%d", "%d"), ("<IENS>", "<ITER>")]
)
def test_that_iens_and_iter_in_runpaths_are_substituted_with_corresponding_indices(
    tmpdir, iens_placeholder, iter_placeholder
):
    with tmpdir.as_cwd():
        ert_config = ErtConfig.from_file_contents(
            f"""
            NUM_REALIZATIONS 10
            RUNPATH simulations/realization-{iens_placeholder}/ITER-{iter_placeholder}
            """
        )
        run_paths = Runpaths(
            jobname_format=ert_config.runpath_config.jobname_format_string,
            runpath_format=ert_config.runpath_config.runpath_format_string,
            filename=".runpath_file",
            substitutions=ert_config.substitutions,
            eclbase=ert_config.runpath_config.eclbase_format_string,
        )
        assert run_paths.get_paths([1, 2, 3], 0) == [
            tmpdir + "/simulations/realization-1/ITER-0",
            tmpdir + "/simulations/realization-2/ITER-0",
            tmpdir + "/simulations/realization-3/ITER-0",
        ]


@pytest.mark.filterwarnings(
    "ignore:.*RUNPATH keyword contains deprecated "
    "value placeholders.*:ert.config.ConfigWarning"
)
@pytest.mark.parametrize("iens_placeholder", [("%d"), ("<IENS>")])
def test_that_runpaths_with_just_iens_will_be_substituted_with_just_iens_index(
    tmpdir, iens_placeholder
):
    with tmpdir.as_cwd():
        ert_config = ErtConfig.from_file_contents(
            f"""
            NUM_REALIZATIONS 10
            RUNPATH simulations/realization-{iens_placeholder}
            """
        )
        run_paths = Runpaths(
            jobname_format=ert_config.runpath_config.jobname_format_string,
            runpath_format=ert_config.runpath_config.runpath_format_string,
            filename=".runpath_file",
            substitutions=ert_config.substitutions,
            eclbase=ert_config.runpath_config.eclbase_format_string,
        )
        assert run_paths.get_paths([1, 2, 3], 0) == [
            tmpdir + "/simulations/realization-1",
            tmpdir + "/simulations/realization-2",
            tmpdir + "/simulations/realization-3",
        ]


@pytest.mark.parametrize(
    "runpath",
    [
        "simulations/realization-<IENS>/iter-%d",
        "simulations/realization-%d/iter-<ITER>",
    ],
)
@pytest.mark.filterwarnings(
    "ignore:.*RUNPATH keyword contains deprecated "
    "value placeholders.*:ert.config.ConfigWarning"
)
def test_that_mixing_printf_format_placeholders_and_bracketed_is_invalid(runpath):
    with pytest.raises(
        ConfigValidationError, match=f"RUNPATH cannot combine deprecated.*{runpath}`."
    ):
        _ = ErtConfig.from_file_contents(
            f"""\
            NUM_REALIZATIONS 1
            RUNPATH {runpath}
            """
        )


@pytest.mark.parametrize(
    "runpath",
    [
        "simulations/realization-<IENS>/iter-<IENS>",
        "simulations/realization-<ITER>/iter-<ITER>",
        "simulations/realization-<IENS>/iter-<ITER>/iter-<ITER>",
    ],
)
def test_that_duplicate_bracketed_placeholders_is_invalid(runpath):
    with pytest.raises(
        ConfigValidationError, match=f"RUNPATH cannot contain multiple.*{runpath}`."
    ):
        _ = ErtConfig.from_file_contents(
            f"""\
            NUM_REALIZATIONS 1
            RUNPATH {runpath}
            """
        )


@pytest.mark.parametrize(
    "runpath",
    [
        "simulations/realization-%d/iter-%d/iter-%d",
        "simulations/realization-%d/realization-%d/iter-%d/iter-%d",
    ],
)
def test_that_more_than_two_printf_format_placeholders_is_invalid(runpath):
    with pytest.raises(
        ConfigValidationError,
        match=f"RUNPATH cannot contain more than two.*{runpath}`.",
    ):
        _ = ErtConfig.from_file_contents(
            f"""\
            NUM_REALIZATIONS 1
            RUNPATH {runpath}
            """
        )


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "placeholder",
    ["<ERTCASE>", "<ERT-CASE>"],
)
def test_that_ertcase_is_replaced_in_runpath(placeholder, make_run_path):
    ert_config = ErtConfig.from_file_contents(
        dedent(
            f"""\
            NUM_REALIZATIONS 1
            JOBNAME a_name_%d
            RUNPATH simulations/{placeholder}/realization-<IENS>/iter-<ITER>
            """
        )
    )
    prior_ensemble, _, _ = make_run_path(ert_config)

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
    for config_node in parameter_configs.values():
        for realization_nr in range(num_realizations):
            if isinstance(config_node, SurfaceConfig):
                prior_ensemble.save_parameters_numpy(
                    np.zeros(dim_size**2).reshape(-1, 1),
                    config_node.name,
                    np.array([realization_nr]),
                )
            elif isinstance(config_node, Field):
                prior_ensemble.save_parameters_numpy(
                    np.zeros(dim_size**3).reshape(-1, 1),
                    config_node.name,
                    np.array([realization_nr]),
                )
            elif isinstance(config_node, GenKwConfig):
                prior_ensemble.save_parameters_numpy(
                    np.zeros(1).reshape(-1, 1),
                    config_node.name,
                    np.array([realization_nr]),
                )
            else:
                raise ValueError(f"unexpected {config_node}")


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize("itr", [0, 1])
def test_when_manifest_files_are_written_loading_succeeds(storage, itr):
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
    Path("gen0.txt").write_text("PARMA0 NORMAL 0 1\n", encoding="utf-8")
    Path("gen1.txt").write_text("PARMA1 NORMAL 0 1\n", encoding="utf-8")
    Path("template.txt").write_text("<PARMA1>", encoding="utf-8")

    config = ErtConfig.from_file_contents(
        dedent(
            f"""\
            NUM_REALIZATIONS {num_realizations}
            DEFINE <ALL> -<ITER>-<IENS>

            RUNPATH simulations/realization-<IENS>/iter-<ITER>

            ECLBASE CASE<ALL>
            GRID GRID.EGRID
            SUMMARY FOPR

            GEN_DATA GENDATA RESULT_FILE:gen_data<ALL>.txt

            SURFACE SURF1 OUTPUT_FILE:surf1_output<ALL>.irap BASE_SURFACE:base.irap \
                FORWARD_INIT:True INIT_FILES:surf1_init<ALL>.irap
            SURFACE SURF2 OUTPUT_FILE:surf2_output<ALL>.irap BASE_SURFACE:base.irap \
                INIT_FILES:%dinit<IENS><ITER>.irap

            FIELD PORO0 PARAMETER field1<ALL>.roff INIT_FILES:field1_init<ALL>.roff \
                FORWARD_INIT:TRUE
            FIELD PORO1 PARAMETER field2<ALL>.roff INIT_FILES:%dinit<IENS><ITER>.roff

            GEN_KW GEN0 gen0.txt
            GEN_KW GEN1 template.txt gen_parameter.txt gen1.txt
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
        jobname_format=config.runpath_config.jobname_format_string,
        runpath_format=config.runpath_config.runpath_format_string,
        filename=str(config.runpath_file),
        substitutions=config.substitutions,
        eclbase=config.runpath_config.eclbase_format_string,
    )

    if itr == 0:
        sample_prior(prior_ensemble, range(num_realizations), 123)
    else:
        save_zeros(prior_ensemble, num_realizations, dim_size=dim_size)

    run_args = create_run_arguments(
        run_paths,
        [True, True],
        prior_ensemble,
    )

    create_run_path(
        run_args=run_args,
        ensemble=prior_ensemble,
        user_config_file=config.user_config_file,
        env_vars=config.env_vars,
        env_pr_fm_step=config.env_pr_fm_step,
        forward_model_steps=config.forward_model_steps,
        substitutions=config.substitutions,
        parameters_file="parameters",
        runpaths=run_paths,
    )

    for i, run_path in enumerate(run_paths.get_paths(range(num_realizations), itr)):
        manifest_path = Path(run_path) / "manifest.json"
        assert manifest_path.exists()
        expected_files = {
            run_path + f"/CASE-{itr}-{i}.UNSMRY",
            run_path + f"/CASE-{itr}-{i}.SMSPEC",
            run_path + f"/gen_data-{itr}-{i}.txt",
        }.union(
            {
                run_path + f"/field1_init-{itr}-{i}.roff",
                run_path + f"/surf1_init-{itr}-{i}.irap",
            }
            if itr == 0
            else set()
        )
        with open(manifest_path, encoding="utf-8") as f:
            manifest = orjson.loads(f.read())
            assert {run_path + "/" + f for f in manifest.values()} == expected_files

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
                raise AssertionError

    # When files in manifest are written we expect loading to succeed
    for run_arg in run_args:
        load_result = asyncio.run(
            load_realization_parameters_and_responses(
                run_arg.runpath, run_arg.iens, run_arg.itr, run_arg.ensemble_storage
            )
        )
        assert load_result.successful
