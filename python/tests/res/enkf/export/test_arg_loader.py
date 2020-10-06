import pytest

from tests import ResTest
from res.test import ErtTestContext

from res.enkf.export import ArgLoader


@pytest.mark.equinor_test
class ArgLoaderTest(ResTest):
    def test_arg_loader(self):

        with self.assertRaises(IOError):
            arg = ArgLoader.load("arg1X")

        arg_file = self.createTestPath(
            "Equinor/config/with_GEN_DATA_RFT/wellpath/WI_1.txt"
        )

        with self.assertRaises(ValueError):
            arg = ArgLoader.load(
                arg_file, column_names=["Col1", "Col2", "Col3", "COl5", "Col6"]
            )

        arg = ArgLoader.load(arg_file, column_names=["utm_x", "utm_y", "md", "tvd"])
        self.assertFloatEqual(arg["utm_x"][0], 461317.620646)
