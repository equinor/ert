import math
import ctypes
import types
import datetime
import time
from   ert.cwrap.cwrap       import *


class ctime(ctypes.c_long):
    def __init__(self , value):
        if isinstance(value , types.IntType):
            self.value = value
        else:
            try:
                self.value = int(math.floor(time.mktime( (value.year , value.month , value.day , value.hour , value.min , value.sec , 0 , 0 , -1 ) )))
            except:
                self.value = int(math.floor(time.mktime( (value.year , value.month , value.day , 0 , 0 , 0 , 0 , 0 , -1 ) )))
                
         
    def ctime(self):
        return self.value

    def time(self):
        """Return this time_t as a time.localtime() object"""
        return time.localtime( self.value )

    def date(self):
        """Return this time_t as a datetime.date([year, month, day])"""
        return datetime.date( *self.time()[0:3])

    def datetime(self):
        return datetime.datetime( *self.time()[0:6] )

    def __str__(self):
        return "%s" % (str(self.datetime()))

    def __ge__(self, other):
        return self.value >= other.value

    def __lt__(self, other):
        return not self >= other



cwrapper = CWrapper( None ) 
cwrapper.registerType( "time_t"  , ctime )
cwrapper.registerType( "time_t*" , ctypes.POINTER(ctime))

