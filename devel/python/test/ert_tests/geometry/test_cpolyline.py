
from ert.geo import CPolyline
from ert.geo.xyz_io import XYZIo
from ert.test import ExtendedTestCase , TestAreaContext


class CPolylineTest(ExtendedTestCase):
    def setUp(self):
        pass

        
    def test_construction(self):
        polyline = CPolyline()
        self.assertEqual( len(polyline) , 0 )
        

    def test_item(self):
        polyline = CPolyline()
        polyline.addPoint( 10 , 20 )
        self.assertEqual( len(polyline) , 1 )

        with self.assertRaises(TypeError):
            (x,y) = polyline["KEY"]
            
        with self.assertRaises(IndexError):
            (x,y) = polyline[10]
            
        (x,y) = polyline[0]
        self.assertEqual( x , 10 )
        self.assertEqual( y , 20 )
        
        polyline.addPoint(20,20)
        (x,y) = polyline[-1]
        self.assertEqual( x , 20 )
        self.assertEqual( y , 20 )
        
        
    def test_cross_segment(self):
        polyline = CPolyline( init_points = [(0,0), (1,0) , (1,1)])
        #
        #            x
        #            |
        #            |
        #            |       
        #    x-------x
        #

        self.assertTrue(polyline.segmentIntersects( (0.5 , 0.5) , (0.5 , -0.5)))
        self.assertTrue(polyline.segmentIntersects( (0.5 , 0.5) , (1.5 , 0.5)))

        self.assertFalse(polyline.segmentIntersects( (0.5 , 0.5) , (0.5 , 1.5)))
        self.assertFalse(polyline.segmentIntersects( (0.5 , 0.5) , (-0.5 , 0.5)))
        
    def test_name(self):
        p1 = CPolyline()
        self.assertTrue( p1.name() is None )

        p2 = CPolyline( name = "Poly2" )
        self.assertEqual( p2.name() , "Poly2")
