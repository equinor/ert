import pytest
from res.enkf import RunContext
from res.enkf.enkf_main import EnKFMain
from res.enkf.res_config import ResConfig


def _create_runpath(enkf_main: EnKFMain) -> RunContext:
    """
    Instantiate an ERT runpath. This will create the parameter coefficients.
    """
    run_context = enkf_main.create_ensemble_experiment_run_context(iteration=0)
    enkf_main.createRunPath(run_context)
    return run_context


def _evaluate_ensemble(enkf_main: EnKFMain, run_context: RunContext):
    """
    Launch ensemble experiment with the created config
    """
    queue_config = enkf_main.get_queue_config()
    job_queue = queue_config.create_job_queue()

    enkf_main.runSimpleStep(job_queue, run_context)


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
        "QUEUE_SYSTEM LOCAL\n"
        "QUEUE_OPTION LOCAL MAX_RUNNING 50\n"
        "NUM_REALIZATIONS 1\n"
        "RUNPATH test/realization-%d/iter-%d\n"
        "INSTALL_JOB dump DUMP\n"
        "FORWARD_MODEL dump\n" + append
    )
    (tmp_path / "DATA").write_text("PARALLEL 8 /")
    (tmp_path / "DUMP").write_text("EXECUTABLE echo\nARGLIST <NUM_CPU>\n")

    config = ResConfig(str(tmp_path / "test.ert"))
    enkf_main = EnKFMain(config)
    run_context = _create_runpath(enkf_main)
    _evaluate_ensemble(enkf_main, run_context)

    with open("test/realization-0/iter-0/dump.stdout.0") as f:
        assert f.read() == f"{numcpu}\n"
