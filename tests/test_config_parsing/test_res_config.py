import os
from datetime import date
from textwrap import dedent

import pytest
from hypothesis import given

from ert._c_wrappers.config.config_parser import ConfigValidationError
from ert._c_wrappers.enkf import ResConfig
from ert._c_wrappers.enkf.config_keys import ConfigKeys

from .config_dict_generator import config_generators, to_config_file


def touch(filename):
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write(" ")


@pytest.mark.xfail(reason="https://github.com/equinor/ert/issues/4178")
def test_res_config_simple_config_parsing(tmpdir, set_site_config, monkeypatch):
    touch(tmpdir + "/rpfile")
    touch(tmpdir + "/datafile")
    os.mkdir(tmpdir + "/license")
    with open(tmpdir + "/test.ert", "w", encoding="utf-8") as fh:
        fh.write(
            """
JOBNAME  Job%d
NUM_REALIZATIONS  1
RUNPATH_FILE rpfile
DATA_FILE datafile
LICENSE_PATH license
"""
        )

    monkeypatch.chdir(tmpdir)
    assert ResConfig("test.ert") == ResConfig(
        config_dict={
            "NUM_REALIZATIONS": 1,
            "DATA_FILE": "datafile",
            "LICENSE_PATH": "license",
            "RES_CONFIG_FILE": "test.ert",
            "RUNPATH_FILE": "rpfile",
        }
    )


def test_res_config_minimal_dict_init():
    config_dict = {ConfigKeys.NUM_REALIZATIONS: 1, ConfigKeys.ENSPATH: "test"}
    res_config = ResConfig(config_dict=config_dict)
    assert res_config is not None


def test_bad_user_config_file_error_message(tmp_path):
    (tmp_path / "test.ert").write_text("NUM_REL 10\n")

    rconfig = None
    with pytest.raises(
        ConfigValidationError, match=r"Parsing.*resulted in the errors:"
    ):
        rconfig = ResConfig(user_config_file=str(tmp_path / "test.ert"))

    assert rconfig is None


def test_bad_config_provide_error_message(tmp_path):
    rconfig = None
    with pytest.raises(ConfigValidationError, match=r"NUM_REALIZATIONS must be set.*"):
        testDict = {"GEN_KW": "a"}
        rconfig = ResConfig(config=testDict)

    assert rconfig is None


@pytest.mark.usefixtures("use_tmpdir")
def test_res_config_parses_date():
    test_config_file_base = "test"
    test_config_file_name = f"{test_config_file_base}.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        DEFINE <STORAGE> storage/<CONFIG_FILE_BASE>-<DATE>
        RUNPATH <STORAGE>/runpath/realization-<IENS>/iter-<ITER>
        ENSPATH <STORAGE>/ensemble
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)
    res_config = ResConfig(user_config_file=test_config_file_name)

    date_string = date.today().isoformat()
    expected_storage = os.path.abspath(f"storage/{test_config_file_base}-{date_string}")
    expected_run_path = f"{expected_storage}/runpath/realization-<IENS>/iter-<ITER>"
    expected_ens_path = f"{expected_storage}/ensemble"
    assert res_config.model_config.ens_path == expected_ens_path
    assert res_config.model_config.runpath_format_string == expected_run_path


@pytest.mark.usefixtures("set_site_config")
@given(config_generators())
def test_env_vars_same_as_from_file(tmp_path_factory, config_generator):
    filename = "config.ert"
    with config_generator(tmp_path_factory, filename) as config_dict:
        assert (
            ResConfig(config_dict=config_dict).env_vars == ResConfig(filename).env_vars
        )


@given(config_generators())
def test_res_config_throws_on_missing_forward_model_job(
    tmp_path_factory, config_generator
):
    filename = "config.ert"
    with config_generator(tmp_path_factory) as config_dict:
        config_dict.pop(ConfigKeys.INSTALL_JOB)
        config_dict.pop(ConfigKeys.INSTALL_JOB_DIRECTORY)
        config_dict[ConfigKeys.FORWARD_MODEL].append(
            {
                ConfigKeys.NAME: "this-is-not-the-job-you-are-looking-for",
                ConfigKeys.ARGLIST: "<WAVE-HAND>=casually",
            }
        )

        to_config_file(filename, config_dict)

        with pytest.raises(expected_exception=ValueError, match="Could not find job"):
            ResConfig(user_config_file=filename)
        with pytest.raises(expected_exception=ValueError, match="Could not find job"):
            ResConfig(config_dict=config_dict)


@pytest.mark.usefixtures("use_tmpdir", "set_site_config")
@pytest.mark.parametrize(
    "bad_define", ["DEFINE A B", "DEFINE <A<B>> C", "DEFINE <A><B> C"]
)
def test_that_non_bracketed_defines_warns(bad_define, capsys):
    with open("test.ert", "w", encoding="utf-8") as fh:
        fh.write(
            dedent(
                f"""
                NUM_REALIZATIONS  1
                {bad_define}
                """
            )
        )

    _ = ResConfig("test.ert")
    assert "Using DEFINE or DATA_KW with substitution" in capsys.readouterr().err


@pytest.mark.usefixtures("set_site_config")
@given(config_generators())
def test_site_config_dict_same_as_from_file(tmp_path_factory, config_generator):
    filename = "config.ert"
    with config_generator(tmp_path_factory, filename) as config_dict:
        assert (
            ResConfig(config_dict=config_dict).env_vars == ResConfig(filename).env_vars
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_substitutions_can_be_done_in_job_names():
    """
    Regression test for a usage case involving setting ECL100 or ECL300
    that was broken by changes to forward_model substitutions.
    """
    test_config_file_name = "test.ert"
    test_config_contents = dedent(
        """
        NUM_REALIZATIONS  1
        DEFINE <ECL100OR300> E100
        FORWARD_MODEL ECLIPS<ECL100OR300>(<VERSION>=1, <NUM_CPU>=42, <OPTS>="-m")
        """
    )
    with open(test_config_file_name, "w", encoding="utf-8") as fh:
        fh.write(test_config_contents)

    res_config = ResConfig(user_config_file=test_config_file_name)
    assert len(res_config.forward_model.jobs) == 1
    job = res_config.forward_model.jobs[0]
    assert job.name == "ECLIPSE100"
