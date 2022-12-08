import json
import os
from pathlib import Path

import pytest

from ert._c_wrappers.enkf import RunContext
from ert._c_wrappers.enkf.enkf_main import EnKFMain
from ert._c_wrappers.enkf.res_config import ResConfig


def _create_runpath(enkf_main: EnKFMain) -> RunContext:
    """
    Instantiate an ERT runpath. This will create the parameter coefficients.
    """
    run_context = enkf_main.create_ensemble_context(
        "prior", [True] * enkf_main.getEnsembleSize(), iteration=0
    )
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

    with open("simulations/realization-0/iter-0/jobs.json", encoding="utf-8") as f:
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
            """,
            encoding="utf-8",
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

        with open("simulations/realization-0/iter-0/jobs.json", encoding="utf-8") as f:
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
