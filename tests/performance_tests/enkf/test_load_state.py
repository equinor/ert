import sys

import pytest

from ert._c_wrappers.enkf import EnKFMain, ResConfig


@pytest.mark.skipif(
    sys.platform.startswith("darwin"),
    reason="https://github.com/equinor/ert/issues/4088",
)
def test_load_from_context(benchmark, template_config):
    with template_config["folder"].as_cwd():
        config = ResConfig("poly.ert")
        ert = EnKFMain(config)
        load_into = ert.getEnkfFsManager().getFileSystem("A1")
        ert.getEnkfFsManager().getFileSystem("default")
        expected_reals = template_config["reals"]
        realisations = [True] * expected_reals
        loaded_reals = benchmark(ert.loadFromForwardModel, realisations, 0, load_into)
        assert loaded_reals == expected_reals


@pytest.mark.skipif(
    sys.platform.startswith("darwin"),
    reason="https://github.com/equinor/ert/issues/4088",
)
def test_load_from_fs(benchmark, template_config):
    with template_config["folder"].as_cwd():
        config = ResConfig("poly.ert")
        ert = EnKFMain(config)
        load_from = ert.getEnkfFsManager().getFileSystem("default")
        expected_reals = template_config["reals"]
        realisations = [True] * expected_reals
        loaded_reals = benchmark(ert.loadFromForwardModel, realisations, 0, load_from)
        assert loaded_reals == expected_reals
