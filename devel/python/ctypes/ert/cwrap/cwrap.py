import math
import ctypes
import types
import re
import datetime
import time
import sys



class ctime(ctypes.c_long):
    def __init__(self , value):
        if isinstance(value , types.IntType):
            self.value = value
        else:
            try:
                self.value = int(math.floor(time.mktime( (value.year , value.month , value.day , value.hour , value.min , value.sec , 0 , 0 , -1 ) )))
            except:
                self.value = int(math.floor(time.mktime( (value.year , value.month , value.day , 0 , 0 , 0 , 0 , 0 , -1 ) )))
                
         

    def time(self):
        """Return this time_t as a time.localtime() object"""
        return time.localtime( self.value )

    def date(self):
        """Return this time_t as a datetime.date([year, month, day])"""
        return datetime.date(*self.time()[0:3])

    def datetime(self):
        return datetime.datetime( *self.time()[0:3] )

    def __str__(self):
        return "%s" % (str(self.datetime()))

    def __ge__(self, other):
        return self.value >= other.value

    def __lt__(self, other):
        return not self >= other




class CWrapper:
    registered_types = {}
    pattern          = re.compile("(?P<return>[a-zA-Z][a-zA-Z0-9_*]*) +(?P<function>[a-zA-Z]\w*) *[(](?P<arguments>[a-zA-Z0-9_*, ]*)[)]")

    def __init__( self , lib ):
        self.lib = lib


    @classmethod
    def registerDefaultTypes(cls):
        """Registers the default available types for prototyping."""
        cls.registerType("void", None)
        cls.registerType("int", ctypes.c_int)
        cls.registerType("int*", ctypes.POINTER(ctypes.c_int))
        cls.registerType("bool", ctypes.c_int)
        cls.registerType("bool*", ctypes.POINTER(ctypes.c_int))
        cls.registerType("long", ctypes.c_long)
        cls.registerType("long*", ctypes.POINTER(ctypes.c_long))
        cls.registerType("char", ctypes.c_char)
        cls.registerType("char*", ctypes.c_char_p)
        cls.registerType("char**", ctypes.POINTER(ctypes.c_char_p))
        cls.registerType("float", ctypes.c_float)
        cls.registerType("float*", ctypes.POINTER(ctypes.c_float))
        cls.registerType("double", ctypes.c_double)
        cls.registerType("double*", ctypes.POINTER(ctypes.c_double))
        cls.registerType("time_t", ctime)
        cls.registerType("time_t*", ctypes.POINTER(ctime))


    @classmethod
    def registerType(cls, type, value ):
        """Register a type against a legal ctypes type"""
        cls.registered_types[type] = value



    def __parseType(self, type):
        """Convert a prototype definition type from string to a ctypes legal type."""
        type = type.strip()

        if CWrapper.registered_types.has_key(type):
            return CWrapper.registered_types[type]
        else:
            return getattr(ctypes , type)

        
    def prototype(self, prototype , lib = None):
        """
        Defines the return type and arguments for a C-function

        prototype expects a string formatted like this:

            "type functionName(type, ... ,type)"

        where type is a type available to ctypes
        Some type are automatically converted:
            int -> c_int
            long -> c_long
            char -> c_char_p
            bool -> c_int
            void -> None
            double -> c_double
            float -> c_float

        There are also pointer versions of these:
            long* -> POINTER(c_long)
            bool* -> POINTER(c_int)
            double* -> POINTER(c_double)
            char* -> c_char_p
            ...
        """

        match = re.match(CWrapper.pattern, prototype)
        if not match:
            sys.stderr.write("Illegal prototype definition: %s\n" % (prototype))
            return None
        else:
            restype = match.groupdict()["return"]
            functioname = match.groupdict()["function"]
            arguments = match.groupdict()["arguments"].split(",")

            func = getattr(self.lib , functioname)
            func.restype = self.__parseType(restype)

            if len(arguments) == 1 and arguments[0].strip() == "":
                func.argtypes = []
            else:
                argtypes = [self.__parseType(arg) for arg in arguments]
                if len(argtypes) == 1 and argtypes[0] is None:
                    argtypes = []
                func.argtypes = argtypes

            return func



class CWrapperNameSpace:
    def __init__( self , name ):
        self.name = name
        
    def __str__(self):
        return "%s wrapper" % self.name

CWrapper.registerDefaultTypes()
