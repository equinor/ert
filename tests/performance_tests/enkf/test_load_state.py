from ert._c_wrappers.enkf import EnKFMain, ResConfig
from ert.storage import open_storage


def test_load_from_context(benchmark, template_config):
    path = template_config["folder"]

    with path.as_cwd(), open_storage(path / "_storage", mode="w") as storage:
        config = ResConfig("poly.ert")
        ert = EnKFMain(config)
        load_into = storage.create_experiment().create_ensemble(
            name="A1", ensemble_size=ert.getEnsembleSize()
        )
        expected_reals = template_config["reals"]
        realisations = [True] * expected_reals
        loaded_reals = benchmark(ert.loadFromForwardModel, realisations, 0, load_into)
        assert loaded_reals == expected_reals


def test_load_from_fs(benchmark, template_config):
    path = template_config["folder"]

    with path.as_cwd(), open_storage(path / "storage", mode="w") as storage:
        config = ResConfig("poly.ert")
        ert = EnKFMain(config)
        load_from = storage.get_ensemble_by_name("default")
        expected_reals = template_config["reals"]
        realisations = [True] * expected_reals
        loaded_reals = benchmark(ert.loadFromForwardModel, realisations, 0, load_from)
        assert loaded_reals == expected_reals
