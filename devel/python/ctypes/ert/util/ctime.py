import time
import datetime
import ctypes



class ctime(ctypes.c_long):
    """A convenience class for working with time_t objects."""

    def time(self):
        """Return this time_t as a time.localtime() object"""
        return time.localtime( self.value )

    def date(self):
        """Return this time_t as a datetime.date([year, month, day])"""
        return datetime.date(*self.time()[0:3])

    def datetime(self):
        return datetime.datetime(*self.time()[0:3] )

    def __str__(self):
        return "%d %s" % (self.value, str(self.datetime()))

    def __ge__(self, other):
        return self.value >= other.value

    def __lt__(self, other):
        return not self >= other

