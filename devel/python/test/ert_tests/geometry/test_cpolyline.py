
from ert.geo import CPolyline
from ert.geo.xyz_io import XYZIo
from ert.test import ExtendedTestCase , TestAreaContext


class CPolylineTest(ExtendedTestCase):
    def setUp(self):
        self.polyline1 = self.createTestPath("local/geometry/pol11.xyz")
        self.polyline2 = self.createTestPath("local/geometry/pol8.xyz")
        self.polyline3 = self.createTestPath("local/geometry/pol8_noend.xyz")
        

        
    def test_construction(self):
        polyline = CPolyline()
        self.assertEqual( len(polyline) , 0 )
        
        with self.assertRaises(IOError):
            CPolyline.createFromXYZFile( "Does/not/exist" )
            
        p1 = CPolyline.createFromXYZFile( self.polyline1 )
        self.assertEqual( len(p1) , 13 )
        x,y = p1[-1]
        self.assertEqual(x , 389789.263184)
        self.assertEqual(y , 6605784.945099)

        p2 = CPolyline.createFromXYZFile( self.polyline2 )
        self.assertEqual( len(p2) , 20 )
        x,y = p2[-1]
        self.assertEqual(x , 396056.314697)
        self.assertEqual(y , 6605835.119461)

        p3 = CPolyline.createFromXYZFile( self.polyline3 )
        self.assertEqual( len(p3) , 20 )
        x,y = p3[-1]
        self.assertEqual(x , 396056.314697)
        self.assertEqual(y , 6605835.119461)


        
    def test_front(self):
        polyline = CPolyline()
        polyline.addPoint( 1 , 1 )
        polyline.addPoint( 0 , 0 , front = True )
        self.assertEqual( len(polyline) , 2 )

        x,y = polyline[0]
        self.assertEqual(x,0)
        self.assertEqual(y,0)

        x,y = polyline[1]
        self.assertEqual(x,1)
        self.assertEqual(y,1)




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
        self.assertTrue( p1.getName() is None )

        p2 = CPolyline( name = "Poly2" )
        self.assertEqual( p2.getName() , "Poly2")
