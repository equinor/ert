import math

from ert.geo import Polyline, GeometryTools , CPolyline
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


    def test_join__polylines(self):
        l1 = Polyline( init_points = [(0,1) , (1,1)])
        l2 = CPolyline( init_points = [(2,-1) , (2,0)])
        l3 = CPolyline( init_points = [(2,2) , (2,3)])
        l4 = Polyline( )
        l5 = CPolyline( init_points = [(0.5,0),(0.5,2)] )

        with self.assertRaises( ValueError ):
            GeometryTools.joinPolylines( l1 , l4 )

        with self.assertRaises( ValueError ):
            GeometryTools.joinPolylines( l4 , l1 )
        
        self.assertIsNone( GeometryTools.joinPolylines( l1 , l5 ))
            
        self.assertEqual( GeometryTools.joinPolylines( l1 , l2 ) , [(1,1) , (2,0)] )
        


