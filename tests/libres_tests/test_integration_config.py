import pytest
from ecl.util.util import BoolVector
from res.enkf import ErtRunContext
from res.enkf.enkf_main import EnKFMain
from res.enkf.res_config import ResConfig


def _create_runpath(enkf_main: EnKFMain) -> ErtRunContext:
    """
    Instantiate an ERT runpath. This will create the parameter coefficients.
    """
    result_fs = enkf_main.getEnkfFsManager().getCurrentFileSystem()

    model_config = enkf_main.getModelConfig()
    runpath_fmt = model_config.getRunpathFormat()
    jobname_fmt = model_config.getJobnameFormat()
    subst_list = enkf_main.getDataKW()

    run_context = ErtRunContext.ensemble_smoother(
        result_fs,
        None,
        BoolVector(default_value=True, initial_size=1),
        runpath_fmt,
        jobname_fmt,
        subst_list,
        0,
    )

    enkf_main.getEnkfSimulationRunner().createRunPath(run_context)
    return run_context


def _evaluate_ensemble(enkf_main: EnKFMain, run_context: ErtRunContext):
    """
    Launch ensemble experiment with the created config
    """
    queue_config = enkf_main.get_queue_config()
    job_queue = queue_config.create_job_queue()

    enkf_main.getEnkfSimulationRunner().runSimpleStep(job_queue, run_context)


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
