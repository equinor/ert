from ert.util.enums import RngAlgTypeEnum, RngInitModeEnum
from ert.util.rng import RandomNumberGenerator
from ert.test import ExtendedTestCase


class RngTest(ExtendedTestCase):

    def test_enums(self):
        self.assertEnumIsFullyDefined(RngAlgTypeEnum, "rng_alg_type", "libert_util/include/ert/util/rng.h")
        self.assertEnumIsFullyDefined(RngInitModeEnum, "rng_init_mode", "libert_util/include/ert/util/rng.h")

    def test_rng_default(self):
        rng = RandomNumberGenerator()
        self.assertIsInstance(rng.getDouble(), float)

        
