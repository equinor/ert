import math

from ert.geo import Polyline, GeometryTools
from ert.geo.xyz_io import XYZIo
from ert.test import ExtendedTestCase , TestAreaContext


class GeometryToolsTest(ExtendedTestCase):

    def test_distance(self):
        p1 = (1,1)
        p2 = (1,2,3)
        with self.assertRaises(ValueError):
            GeometryTools.distance( p1 , p2)

        with self.assertRaises(TypeError):
            GeometryTools.distance( 1 , p2 )

        p2 = (2,2)
        self.assertEqual( GeometryTools.distance( p1 , p2) , math.sqrt(2))

        p1 = (1,1,1)
        p2 = (2,2,2)
        self.assertEqual( GeometryTools.distance( p1 , p2) , math.sqrt(3))
