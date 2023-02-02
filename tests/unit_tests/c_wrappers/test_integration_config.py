import json
import os
from argparse import ArgumentParser
from pathlib import Path

import pytest

from ert.__main__ import ert_parser
from ert._c_wrappers.enkf import RunContext
from ert._c_wrappers.enkf.enkf_main import EnKFMain
from ert._c_wrappers.enkf.res_config import ResConfig
from ert.cli import TEST_RUN_MODE
from ert.cli.main import run_cli


def _create_runpath(enkf_main: EnKFMain) -> RunContext:
    """
    Instantiate an ERT runpath. This will create the parameter coefficients.
    """
    run_context = enkf_main.create_ensemble_experiment_run_context(iteration=0)
    enkf_main.createRunPath(run_context)
    return run_context


@pytest.mark.parametrize(
    "append,numcpu",
    [
        ("", 1),  # Default is 1
        ("NUM_CPU 2\n", 2),
        ("DATA_FILE DATA\n", 8),  # Data file dictates NUM_CPU with PARALLEL
        ("NUM_CPU 3\nDATA_FILE DATA\n", 3),  # Explicit NUM_CPU supersedes PARALLEL
    ],
)
def test_num_cpu_subst(monkeypatch, tmp_path, append, numcpu):
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

    config = ResConfig(str(tmp_path / "test.ert"))
    enkf_main = EnKFMain(config)
    _create_runpath(enkf_main)

    with open("simulations/realization-0/iter-0/jobs.json") as f:
        assert f'"argList": ["{numcpu}"]' in f.read()


def test_environment_var_list(tmpdir):
    with tmpdir.as_cwd():
        Path("test.ert").write_text(
            """
            NUM_REALIZATIONS 1
            SETENV FIRST $PATH
            SETENV SECOND $MYVAR
            SETENV THIRD  TheThirdValue
            UPDATE_PATH   FOURTH TheFourthValue
            """
        )
        my_var_text = "THIS_IS_MY_VAR"
        os.environ["MYVAR"] = my_var_text
        path_env_var = os.getenv("PATH")
        try:
            config = ResConfig("test.ert")
            enkf_main = EnKFMain(config)
            _create_runpath(enkf_main)
        except Exception as e:
            raise e
        finally:
            del os.environ["MYVAR"]

        with open("simulations/realization-0/iter-0/jobs.json") as f:
            data = json.load(f)
            global_env = data.get("global_environment")
            global_update_path = data.get("global_update_path")

        assert global_env["FIRST"] == path_env_var
        assert config.env_vars.varlist["FIRST"] == path_env_var
        assert global_env["SECOND"] == my_var_text
        assert config.env_vars.varlist["SECOND"] == my_var_text
        assert global_env["THIRD"] == "TheThirdValue"
        assert config.env_vars.varlist["THIRD"] == "TheThirdValue"

        assert global_update_path["FOURTH"] == "TheFourthValue"
        assert config.env_vars.updatelist["FOURTH"] == "TheFourthValue"


def replace_string_in_file(file: str, from_string: str, to_string: str):
    with open(file, encoding="utf-8") as f:
        content = f.read()

    with open(file, mode="w", encoding="utf-8") as f:
        f.write(content.replace(from_string, to_string))


def test_that_outfile_from_gen_kw_template_creates_relative_path(copy_case):
    copy_case("poly_example")

    replace_string_in_file("poly.ert", "coeffs.json", "somepath/coeffs.json")
    replace_string_in_file("poly_eval.py", "coeffs.json", "somepath/coeffs.json")

    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(
        parser,
        [
            TEST_RUN_MODE,
            "poly.ert",
            "--port-range",
            "1024-65535",
        ],
    )
    run_cli(parsed)

    assert os.path.exists("poly_out/realization-0/iter-0/somepath/coeffs.json")


# This test was added to show current behaviour for Ert.
# If absolute paths should be possible to be used like this is up for debate.
def test_that_outfile_from_gen_kw_template_accepts_absolute_path(copy_case):
    copy_case("poly_example")

    replace_string_in_file("poly.ert", "coeffs.json", "/tmp/somepath/coeffs.json")
    replace_string_in_file("poly_eval.py", "coeffs.json", "tmp/somepath/coeffs.json")

    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(
        parser,
        [
            TEST_RUN_MODE,
            "poly.ert",
            "--port-range",
            "1024-65535",
        ],
    )
    run_cli(parsed)

    assert os.path.exists("poly_out/realization-0/iter-0/tmp/somepath/coeffs.json")
