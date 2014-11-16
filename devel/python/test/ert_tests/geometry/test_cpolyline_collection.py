
from ert.geo import CPolylineCollection , CPolyline
from ert.geo.xyz_io import XYZIo
from ert.test import ExtendedTestCase , TestAreaContext


class CPolylineCollectionTest(ExtendedTestCase):

    def test_construction(self):
        pc = CPolylineCollection()
        self.assertEqual(len(pc) , 0)
        

    def test_add_polyline(self):
        pc = CPolylineCollection()
        pl = pc.createPolyline( name = "TestP" )
        self.assertTrue( isinstance(pl , CPolyline))
        self.assertEqual(len(pc) , 1)
        self.assertTrue( "TestP" in pc )
        
        with self.assertRaises(IndexError):
            pl = pc[2]

        p0 = pc[0]
        self.assertTrue( p0 == pl )
            
        with self.assertRaises(KeyError):
            pn = pc["missing"]
            
        pn = pc["TestP"]
        self.assertTrue( pn == pl )

        px = CPolyline( name = "TestP")
        with self.assertRaises(KeyError):
            pc.addPolyline( px )
        self.assertEqual(len(pc) , 1)
            

        p2 = CPolyline( name = "Poly2")
        pc.addPolyline( p2 )
        
        self.assertEqual( len(pc) , 2 )
        self.assertTrue( "Poly2" in pc )

        l = []
        for p in pc:
            l.append(p)
        self.assertEqual( len(pc) , 2 )

        
