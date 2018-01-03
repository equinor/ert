from tests import ResTest
from res.test import ErtTestContext

from ecl.util import BoolVector,IntVector
from res.enkf import ActiveMode, EnsembleConfig
from res.enkf import ObsVector , LocalObsdata


class EnsembleConfigTest(ResTest):

    def test_create(self):
        conf = EnsembleConfig( )
        self.assertEqual( len(conf) , 0 )
        self.assertFalse( "XYZ" in conf )

        with self.assertRaises(KeyError):
            node = conf["KEY"]
