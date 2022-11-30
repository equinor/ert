"""
These tests currently only check existing behavior so to ensure backwards compatability.
"""
import pytest

from ert._c_wrappers.enkf import EnKFMain, ResConfig


def read_jobname(config_file):
    res_config = ResConfig(config_file)
    ert = EnKFMain(res_config)
    run_context = ert.create_ensemble_experiment_run_context(0)
    ert.createRunPath(run_context)
    return run_context[0].job_name


@pytest.mark.parametrize(
    "defines, expected",
    [
        pytest.param(
            """
NUM_REALIZATIONS 1
DEFINE <A> <B><C>
DEFINE <B> my
DEFINE <C> name
JOBNAME <A>%d""",
            "myname0",
            id="Testing of use before declaration works for defines",
        ),
        pytest.param(
            """
NUM_REALIZATIONS 1
DATA_KW <A> <B><C>
DATA_KW <B> my
DATA_KW <C> name
JOBNAME <A>%d""",
            "myname0",
            id="Testing of use before declaration works for data_kw",
        ),
        pytest.param(
            """
NUM_REALIZATIONS 1
DEFINE <B> my
DEFINE <C> name
DEFINE <A> <B><C>
JOBNAME <A>%d""",
            "myname0",
            id="Testing of declaration before use works for defines",
        ),
        pytest.param(
            """
NUM_REALIZATIONS 1
DATA_KW <B> my
DATA_KW <C> name
DATA_KW <A> <B><C>
JOBNAME <A>%d""",
            "<B><C>0",
            id="Testing of declaration before use fails for data_kw",
        ),
    ],
)
def test_some_stuff(defines, expected, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "config_file.ert"
    config_file.write_text(defines)

    assert read_jobname(str(config_file)) == expected
