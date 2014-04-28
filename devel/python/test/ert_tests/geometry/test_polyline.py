from ert.geo import Polyline
from ert.geo.xyz_reader import XYZReader
from ert.test.extended_testcase import ExtendedTestCase


class PolylineTest(ExtendedTestCase):
    def setUp(self):
        self.polyline = self.createTestPath("local/geometry/pol11.xyz")
        self.closed_polyline = self.createTestPath("local/geometry/pol8.xyz")

    def test_construction(self):
        polyline = Polyline(name="test line")

        with self.assertRaises(IndexError):
            polyline.isClosed()

        self.assertEqual(polyline.name(), "test line")

        self.assertEqual(len(polyline), 0)

        polyline.addPoint(0, 0, 0)
        self.assertEqual(len(polyline), 1)

        polyline.addPoint(1, 1, 0)
        self.assertEqual(len(polyline), 2)

        self.assertEqual(polyline[0], (0, 0, 0))
        self.assertEqual(polyline[1], (1, 1, 0))

        polyline.addPoint(0, 1, 0)
        self.assertFalse(polyline.isClosed())

        polyline.addPoint(0, 0, 0)
        self.assertTrue(polyline.isClosed())


    def test_iteration(self):
        values = [(0, 0, 0),
                  (1, 0, 0),
                  (1, 1, 0),
                  (1, 1, 1)]

        polyline = Polyline(name="iteration line")

        for p in values:
            polyline.addPoint(*p)

        for index, point in enumerate(polyline):
            self.assertEqual(point, values[index])



    def test_read_xyz_from_file(self):
        with self.assertRaises(ValueError):
            XYZReader.readXYZFile("does/not/exist.xyz")

        polyline = XYZReader.readXYZFile(self.polyline)

        self.assertEqual(polyline.name(), "pol11.xyz")
        self.assertEqual(len(polyline), 13)
        self.assertFalse(polyline.isClosed())
        self.assertEqual(polyline[0], (390271.843750, 6606121.334396, 1441.942627))  # first point
        self.assertEqual(polyline[12], (389789.263184, 6605784.945099, 1446.627808))  # last point

        polyline = XYZReader.readXYZFile(self.closed_polyline)

        self.assertEqual(polyline.name(), "pol8.xyz")
        self.assertEqual(len(polyline), 21)
        self.assertTrue(polyline.isClosed())
        self.assertEqual(polyline[0], (396202.413086, 6606091.935028, 1542.620972))  # first point
        self.assertEqual(polyline[20], (396202.413086, 6606091.935028, 1542.620972))  # last point







