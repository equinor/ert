import res.enkf
from res.enkf.enkf_main import EnKFMain
from res.enkf.substituter import Substituter
from res.enkf.runpaths import Runpaths
from unittest.mock import MagicMock


class MockEnKFMain(EnKFMain):
    def __init__(self):
        self._real_enkf_main = MagicMock()
        self._real_enkf_main.return_value = MagicMock()
        self._real_enkf_main.return_value.substituter = Substituter()
        self._real_enkf_main.return_value.runpaths = Runpaths("job%d", "/run%d/path%d")


def test_load_from_forward_model():
    enkf_main = MockEnKFMain()
    fs = MagicMock()
    realizations = [True] * 10
    iteration = 0
    num_loaded = 8

    enkf_main.create_ensemble_experiment_run_context = MagicMock()
    enkf_main.loadFromRunContext = MagicMock()
    enkf_main.loadFromRunContext.return_value = num_loaded

    assert enkf_main.loadFromForwardModel(realizations, iteration, fs) == num_loaded

    enkf_main.loadFromRunContext.assert_called()


def test_create_ensemble_experiment_run_context():
    enkf_main = MockEnKFMain()
    fs = MagicMock()

    enkf_main._create_run_context = MagicMock()

    realizations = [True] * 10
    iteration = 0

    enkf_main.create_ensemble_experiment_run_context(
        active_mask=realizations, source_filesystem=fs, iteration=iteration
    )

    enkf_main._create_run_context.assert_called_with(
        iteration=iteration,
        active_mask=realizations,
        source_filesystem=fs,
        target_fs=None,
    )


def test_create_ensemble_smoother_run_context():
    enkf_main = MockEnKFMain()
    fs = MagicMock()
    fs2 = MagicMock()

    enkf_main._create_run_context = MagicMock()

    realizations = [True] * 10
    iteration = 0

    enkf_main.create_ensemble_smoother_run_context(
        active_mask=realizations,
        source_filesystem=fs,
        target_filesystem=fs2,
        iteration=iteration,
    )

    enkf_main._create_run_context.assert_called_with(
        iteration=iteration,
        active_mask=realizations,
        source_filesystem=fs,
        target_fs=fs2,
    )


def test_create_run_context(monkeypatch):
    enkf_main = MockEnKFMain()

    iteration = 0
    ensemble_size = 10
    run_context = MagicMock()
    monkeypatch.setattr(res.enkf.enkf_main, "RunContext", run_context)
    enkf_main.getEnsembleSize = MagicMock()
    enkf_main.getEnsembleSize.return_value = ensemble_size
    enkf_main.getEnkfFsManager = MagicMock()

    enkf_main._create_run_context(
        iteration=iteration,
    )
    run_context.assert_called_with(
        sim_fs=enkf_main.getEnkfFsManager().getCurrentFileSystem.return_value,
        target_fs=None,
        mask=[True] * ensemble_size,
        iteration=iteration,
        paths=enkf_main.runpaths.get_paths(list(range(ensemble_size)), iteration),
        jobnames=enkf_main.runpaths.get_jobnames(list(range(ensemble_size)), iteration),
    )

    assert enkf_main.substituter.get_substitutions(1, iteration) == {
        "<RUNPATH>": f"/run1/path{iteration}",
        "<ECL_BASE>": "job1",
        "<ECLBASE>": "job1",
        "<ITER>": str(iteration),
        "<IENS>": "1",
    }


def test_create_set_geo_id():
    enkf_main = MockEnKFMain()

    iteration = 1
    realization = 2
    geo_id = "geo_id"

    enkf_main.set_geo_id("geo_id", realization, iteration)

    assert (
        enkf_main.substituter.get_substitutions(realization, iteration)["<GEO_ID>"]
        == geo_id
    )
