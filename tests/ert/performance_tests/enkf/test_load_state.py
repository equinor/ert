from ert import LibresFacade
from ert.config import ErtConfig
from ert.storage import open_storage


def test_load_from_context(benchmark, template_config):
    path = template_config["folder"]

    with path.as_cwd(), open_storage(path / "_storage", mode="w") as storage:
        config = ErtConfig.from_file("poly.ert")
        load_into = storage.create_experiment().create_ensemble(
            name="A1", ensemble_size=config.runpath_config.num_realizations
        )
        expected_reals = template_config["reals"]
        loaded_reals = benchmark(
            LibresFacade.load_from_run_path,
            config.runpath_config.runpath_format_string,
            load_into,
            list(range(expected_reals)),
        )
        assert loaded_reals == expected_reals
