import pytest

from res.test import ErtTestSharedContext


@pytest.mark.equinor_test
@pytest.mark.benchmark
@pytest.mark.usefixtures("class_source_root")
class TestLoadState:
    def test_load_from_context(self, benchmark):
        with ErtTestSharedContext(
            self.TESTDATA_ROOT / "Equinor/config/with_data/config_GEN_DATA"
        ) as ctx:
            load_into = ctx.ert.getEnkfFsManager().getFileSystem("A1")
            ctx.ert.getEnkfFsManager().getFileSystem("default")

            realisations = [True] * 25

            run_context = ctx.ert.getRunContextENSEMPLE_EXPERIMENT(
                load_into, realisations
            )

            loaded = benchmark(ctx.ert.loadFromRunContext, run_context, load_into)
            assert loaded == 25, f"Loaded {loaded} realizations, expected 25"

    def test_load_from_fs(self, benchmark):
        with ErtTestSharedContext(
            self.TESTDATA_ROOT / "Equinor/config/with_data/config_GEN_DATA"
        ) as ctx:
            load_from = ctx.ert.getEnkfFsManager().getFileSystem("default")
            realisations = [True] * 25
            loaded = benchmark(ctx.ert.loadFromForwardModel, realisations, 0, load_from)
            assert loaded == 25, f"Loaded {loaded} realizations, expected 25"
