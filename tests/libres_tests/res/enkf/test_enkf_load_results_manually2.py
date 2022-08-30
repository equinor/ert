import logging

import pytest

from ert._c_wrappers.enkf import EnKFMain


@pytest.mark.skip
@pytest.mark.parametrize("lazy_load", [True, False])
def test_load_results_manually2(setup_case, caplog, monkeypatch, lazy_load):
    """
    This little test does not depend on Equinor-data and only verifies
    the lazy_load flag in forward_load_context plus memory-logging
    """
    if lazy_load:
        monkeypatch.setenv("ERT_LAZY_LOAD_SUMMARYDATA", str(lazy_load))
    res_config = setup_case("local/snake_oil", "snake_oil.ert")
    ert = EnKFMain(res_config)
    load_from = ert.getEnkfFsManager().getFileSystem("default_0")
    ert.getEnkfFsManager().switchFileSystem(load_from)
    realisations = [False] * 25
    realisations[0] = True  # only need one to test what we want
    with caplog.at_level(logging.INFO):
        loaded = ert.loadFromForwardModel(realisations, 0, load_from)
        assert 0 == loaded  # they will in fact all fail, but that's ok
        assert f"lazy={lazy_load}".lower() in caplog.text
