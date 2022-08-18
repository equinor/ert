import pytest
from ert._c_wrappers.enkf import RunContext
from ert._c_wrappers.enkf.enkf_main import EnKFMain
from ert._c_wrappers.enkf.res_config import ResConfig


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

    with open("simulations/realization0/jobs.json") as f:
        assert f'"argList" : ["{numcpu}"]' in f.read()
