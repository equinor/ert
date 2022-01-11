import pytest
import py
from pandas.core.frame import DataFrame
from libres_utils import ResTest
from res.enkf.export import GenDataCollector, GenKwCollector
from res.test import ErtTestContext
from ert_shared.cli import ENSEMBLE_SMOOTHER_MODE
from ert_shared.cli.main import run_cli
from ert_shared.main import ert_parser
from argparse import ArgumentParser


class TestPerformance(ResTest):
    def setUp(self):
        self.poly_example_config = self.createTestPath("local/poly_example/poly.ert")
        self.snake_oil_config = self.createTestPath("local/snake_oil/snake_oil.ert")

    def tearDown(self):
        pass

    @pytest.fixture(autouse=True)
    def setupBenchmark(self, benchmark):
        self.benchmark = benchmark

    def run_poly_example(self, poly_dir: py.path) -> None:
        with poly_dir.as_cwd():
            parser = ArgumentParser(prog="test_main")
            parsed = ert_parser(
                parser,
                [
                    ENSEMBLE_SMOOTHER_MODE,
                    "--target-case",
                    "poly_runpath_file",
                    "--realizations",
                    "1,2,4",
                    "poly.ert",
                    "--port-range",
                    "1024-65535",
                ],
            )
            run_cli(parsed)

    def test_gen_data_collector_snake_oil(self):
        with ErtTestContext(
            "python/enkf/export/gen_data_collector", self.snake_oil_config
        ) as context:
            ert = context.getErt()

            df: DataFrame = self.benchmark(
                GenDataCollector.loadGenData,
                ert,
                "default_0",
                "SNAKE_OIL_OPR_DIFF",
                199,
            )

            print(df.head(10))
            print(df.info(memory_usage="deep"))

    def test_gen_kw_collector_snake_oil(self):
        with ErtTestContext(
            "python/enkf/export/gen_kw_collector", self.snake_oil_config
        ) as context:
            ert = context.getErt()

            df: DataFrame = self.benchmark(
                GenKwCollector.loadAllGenKwData, ert, "default_0"
            )

            print(df.head(10))
            print(df.info(memory_usage="deep"))

    def test_gen_data_collector_poly_example(self):
        with ErtTestContext(
            "python/enkf/export/gen_data_collector", self.poly_example_config
        ) as context:
            ert = context.getErt()

            self.run_poly_example(py.path.local(context._tmp_dir))

            df: DataFrame = self.benchmark(
                GenDataCollector.loadGenData, ert, "default", "POLY_RES", 0
            )

            print(df.head(10))
            print(df.info(memory_usage="deep"))

    def test_gen_kw_collector_poly_example(self):
        with ErtTestContext(
            "python/enkf/export/gen_kw_collector", self.poly_example_config
        ) as context:
            ert = context.getErt()

            self.run_poly_example(py.path.local(context._tmp_dir))

            df: DataFrame = self.benchmark(
                GenKwCollector.loadAllGenKwData, ert, "default"
            )

            print(df.head(10))
            print(df.info(memory_usage="deep"))
