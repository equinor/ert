from ert import LibresFacade
from ert.config import ErtConfig
from ert.storage import open_storage


def test_load_from_context(benchmark, template_config):
    path = template_config["folder"]

    with path.as_cwd(), open_storage(path / "_storage", mode="w") as storage:
        ert = LibresFacade.from_config_file("poly.ert")
        load_into = storage.create_experiment().create_ensemble(
            name="A1", ensemble_size=ert.get_ensemble_size()
        )
        expected_reals = template_config["reals"]
        realisations = [True] * expected_reals
        loaded_reals = benchmark(
            ert.load_from_forward_model, load_into, realisations, 0
        )
        assert loaded_reals == expected_reals


def test_load_from_fs(benchmark, template_config):
    path = template_config["folder"]

    with path.as_cwd(), open_storage(path / "storage", mode="w") as storage:
        ert = LibresFacade.from_config_file("poly.ert")
        load_from = storage.get_ensemble_by_name("default")
        expected_reals = template_config["reals"]
        realisations = [True] * expected_reals
        loaded_reals = benchmark(
            ert.load_from_forward_model, load_from, realisations, 0
        )
        assert loaded_reals == expected_reals
