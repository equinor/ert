from ert._c_wrappers.enkf import EnKFMain, ErtConfig


def test_load_from_context(benchmark, template_config):
    with template_config["folder"].as_cwd():
        config = ErtConfig.from_file("poly.ert")
        ert = EnKFMain(config)
        load_into = ert.storage_manager.add_case("A1")
        expected_reals = template_config["reals"]
        realisations = [True] * expected_reals
        loaded_reals = benchmark(ert.loadFromForwardModel, realisations, 0, load_into)
        assert loaded_reals == expected_reals


def test_load_from_fs(benchmark, template_config):
    with template_config["folder"].as_cwd():
        config = ErtConfig.from_file("poly.ert")
        ert = EnKFMain(config)
        load_from = ert.storage_manager["default"]
        expected_reals = template_config["reals"]
        realisations = [True] * expected_reals
        loaded_reals = benchmark(ert.loadFromForwardModel, realisations, 0, load_from)
        assert loaded_reals == expected_reals
