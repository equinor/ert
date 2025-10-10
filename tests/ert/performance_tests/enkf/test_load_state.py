import pytest

from ert.config import ErtConfig
from ert.storage import open_storage
from ert.storage.local_ensemble import load_parameters_and_responses_from_runpath


@pytest.mark.filterwarnings("ignore:Config contains a SUMMARY key but no forward")
def test_load_from_context(benchmark, template_config):
    path = template_config["folder"]

    with path.as_cwd(), open_storage(path / "_storage", mode="w") as storage:
        config = ErtConfig.from_file("poly.ert")
        load_into = storage.create_experiment().create_ensemble(
            name="A1", ensemble_size=config.runpath_config.num_realizations
        )
        expected_reals = template_config["reals"]
        loaded_reals = benchmark(
            load_parameters_and_responses_from_runpath,
            config.runpath_config.runpath_format_string,
            load_into,
            list(range(expected_reals)),
        )
        assert loaded_reals == expected_reals
