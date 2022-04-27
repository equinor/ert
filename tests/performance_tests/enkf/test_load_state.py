from res.enkf import ResConfig, EnKFMain


def test_load_from_context(benchmark, template_config):
    with template_config["folder"].as_cwd():
        config = ResConfig("poly.ert")
        ert = EnKFMain(config, strict=True)
        load_into = ert.getEnkfFsManager().getFileSystem("A1")
        ert.getEnkfFsManager().getFileSystem("default")
        expected_reals = template_config["reals"]
        realisations = [True] * expected_reals
        run_context = ert.getRunContextENSEMPLE_EXPERIMENT(load_into, realisations)
        loaded_reals = benchmark(ert.loadFromRunContext, run_context, load_into)
        assert loaded_reals == expected_reals


def test_load_from_fs(benchmark, template_config):
    with template_config["folder"].as_cwd():
        config = ResConfig("poly.ert")
        ert = EnKFMain(config, strict=True)
        load_from = ert.getEnkfFsManager().getFileSystem("default")
        expected_reals = template_config["reals"]
        realisations = [True] * expected_reals
        loaded_reals = benchmark(ert.loadFromForwardModel, realisations, 0, load_from)
        assert loaded_reals == expected_reals
