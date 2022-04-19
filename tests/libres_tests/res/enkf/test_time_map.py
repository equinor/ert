import datetime

from ecl.util.test import TestAreaContext
from libres_utils import ResTest

from res.enkf import TimeMap


class TimeMapTest(ResTest):
    def test_time_map(self):
        with self.assertRaises(IOError):
            TimeMap("Does/not/exist")

        tm = TimeMap()
        with self.assertRaises(IndexError):
            tm[10]

        pfx = "TimeMap("
        rep = repr(tm)
        print('repr(time_map) = "%s"' % repr(tm))
        self.assertEqual(pfx, rep[: len(pfx)])

        self.assertTrue(tm.update(0, datetime.date(2000, 1, 1)))
        self.assertEqual(tm[0], datetime.date(2000, 1, 1))

        with self.assertRaises(Exception):
            tm.update(0, datetime.date(2000, 1, 2))

        self.assertTrue(tm.update(1, datetime.date(2000, 1, 2)))
        d = tm.dump()
        self.assertEqual(
            d, [(0, datetime.date(2000, 1, 1), 0), (1, datetime.date(2000, 1, 2), 1)]
        )

    def test_fscanf(self):
        tm = TimeMap()

        with self.assertRaises(IOError):
            tm.fload("Does/not/exist")

        with TestAreaContext("timemap/fload1"):
            with open("map.txt", "w") as fileH:
                fileH.write("2000-10-10\n")
                fileH.write("2000-10-12\n")
                fileH.write("2000-10-14\n")
                fileH.write("2000-10-16\n")

            tm.fload("map.txt")
            self.assertEqual(4, len(tm))
            self.assertEqual(datetime.date(2000, 10, 10), tm[0])
            self.assertEqual(datetime.date(2000, 10, 16), tm[3])

        with TestAreaContext("timemap/fload2"):
            with open("map.txt", "w") as fileH:
                fileH.write("10/10/2000\n")
                fileH.write("12/10/2000\n")
                fileH.write("14/10/2000\n")
                fileH.write("16/10/2000\n")

            tm.fload("map.txt")
            self.assertEqual(4, len(tm))
            self.assertEqual(datetime.date(2000, 10, 10), tm[0])
            self.assertEqual(datetime.date(2000, 10, 16), tm[3])

        with TestAreaContext("timemap/fload3"):
            with open("map.txt", "w") as fileH:
                fileH.write("10/10/200X\n")

            with self.assertRaises(Exception):
                tm.fload("map.txt")

            self.assertEqual(4, len(tm))
            self.assertEqual(datetime.date(2000, 10, 10), tm[0])
            self.assertEqual(datetime.date(2000, 10, 16), tm[3])

        with TestAreaContext("timemap/fload2"):
            with open("map.txt", "w") as fileH:
                fileH.write("12/10/2000\n")
                fileH.write("10/10/2000\n")

            with self.assertRaises(Exception):
                tm.fload("map.txt")

            self.assertEqual(4, len(tm))
            self.assertEqual(datetime.date(2000, 10, 10), tm[0])
            self.assertEqual(datetime.date(2000, 10, 16), tm[3])

    def test_setitem(self):
        tm = TimeMap()
        tm[0] = datetime.date(2000, 1, 1)
        tm[1] = datetime.date(2000, 1, 2)
        self.assertEqual(2, len(tm))

        self.assertEqual(tm[0], datetime.date(2000, 1, 1))
        self.assertEqual(tm[1], datetime.date(2000, 1, 2))

    def test_in(self):
        tm = TimeMap()
        tm[0] = datetime.date(2000, 1, 1)
        tm[1] = datetime.date(2000, 1, 2)
        tm[2] = datetime.date(2000, 1, 3)

        self.assertTrue(datetime.date(2000, 1, 1) in tm)
        self.assertTrue(datetime.date(2000, 1, 2) in tm)
        self.assertTrue(datetime.date(2000, 1, 3) in tm)

        self.assertFalse(datetime.date(2001, 1, 3) in tm)
        self.assertFalse(datetime.date(1999, 1, 3) in tm)

    def test_lookupDate(self):
        tm = TimeMap()
        tm[0] = datetime.date(2000, 1, 1)
        tm[1] = datetime.date(2000, 1, 2)
        tm[2] = datetime.date(2000, 1, 3)

        self.assertEqual(0, tm.lookupTime(datetime.date(2000, 1, 1)))
        self.assertEqual(0, tm.lookupTime(datetime.datetime(2000, 1, 1, 0, 0, 0)))

        self.assertEqual(2, tm.lookupTime(datetime.date(2000, 1, 3)))
        self.assertEqual(2, tm.lookupTime(datetime.datetime(2000, 1, 3, 0, 0, 0)))

        with self.assertRaises(ValueError):
            tm.lookupTime(datetime.date(1999, 10, 10))

    def test_lookupDays(self):
        tm = TimeMap()

        with self.assertRaises(ValueError):
            tm.lookupDays(0)

        tm[0] = datetime.date(2000, 1, 1)
        tm[1] = datetime.date(2000, 1, 2)
        tm[2] = datetime.date(2000, 1, 3)

        self.assertEqual(0, tm.lookupDays(0))
        self.assertEqual(1, tm.lookupDays(1))
        self.assertEqual(2, tm.lookupDays(2))

        with self.assertRaises(ValueError):
            tm.lookupDays(-1)

        with self.assertRaises(ValueError):
            tm.lookupDays(0.50)

        with self.assertRaises(ValueError):
            tm.lookupDays(3)

    def test_nearest_date_lookup(self):
        tm = TimeMap()
        with self.assertRaises(ValueError):
            tm.lookupTime(datetime.date(1999, 1, 1))

        with self.assertRaises(ValueError):
            tm.lookupTime(
                datetime.date(1999, 1, 1),
                tolerance_seconds_before=10,
                tolerance_seconds_after=10,
            )

        tm[0] = datetime.date(2000, 1, 1)
        tm[1] = datetime.date(2000, 2, 1)
        tm[2] = datetime.date(2000, 3, 1)

        # Outside of total range will raise an exception, irrespective of
        # the tolerances used.
        with self.assertRaises(ValueError):
            tm.lookupTime(
                datetime.date(1999, 1, 1),
                tolerance_seconds_before=-1,
                tolerance_seconds_after=-1,
            )

        with self.assertRaises(ValueError):
            tm.lookupTime(
                datetime.date(2001, 1, 1),
                tolerance_seconds_before=-1,
                tolerance_seconds_after=-1,
            )

        self.assertEqual(
            0,
            tm.lookupTime(
                datetime.datetime(2000, 1, 1, 0, 0, 10), tolerance_seconds_after=15
            ),
        )
        self.assertEqual(
            1,
            tm.lookupTime(
                datetime.datetime(2000, 1, 1, 0, 0, 10),
                tolerance_seconds_before=3600 * 24 * 40,
            ),
        )

        self.assertEqual(
            0,
            tm.lookupTime(
                datetime.date(2000, 1, 10),
                tolerance_seconds_before=-1,
                tolerance_seconds_after=-1,
            ),
        )
        self.assertEqual(
            1,
            tm.lookupTime(
                datetime.date(2000, 1, 20),
                tolerance_seconds_before=-1,
                tolerance_seconds_after=-1,
            ),
        )

        with self.assertRaises(ValueError):
            tm.lookupTime(
                datetime.date(2001, 10, 1),
                tolerance_seconds_before=10,
                tolerance_seconds_after=10,
            )
