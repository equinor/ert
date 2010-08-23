import ctypes
import re
import datetime
import time
import sys


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
        return "%s" % (str(self.datetime()))

    def __ge__(self, other):
        return self.value >= other.value

    def __lt__(self, other):
        return not self >= other




class CWrapper:
    def __init__( self , lib ):
        self.lib = lib
        self.pattern = re.compile("(?P<return>[a-zA-Z][a-zA-Z0-9_*]*) +(?P<function>[a-zA-Z]\w*) *[(](?P<arguments>[a-zA-Z0-9_*, ]*)[)]")
        self.__registerDefaultTypes()


    def __registerDefaultTypes(self):
        """Registers the default available types for prototyping."""
        self.registered_types = {}
        self.registerType("void", None)
        self.registerType("int", ctypes.c_int)
        self.registerType("int*", ctypes.POINTER(ctypes.c_int))
        self.registerType("bool", ctypes.c_int)
        self.registerType("bool*", ctypes.POINTER(ctypes.c_int))
        self.registerType("long", ctypes.c_long)
        self.registerType("long*", ctypes.POINTER(ctypes.c_long))
        self.registerType("char", ctypes.c_char)
        self.registerType("char*", ctypes.c_char_p)
        self.registerType("char**", ctypes.POINTER(ctypes.c_char_p))
        self.registerType("float", ctypes.c_float)
        self.registerType("float*", ctypes.POINTER(ctypes.c_float))
        self.registerType("double", ctypes.c_double)
        self.registerType("double*", ctypes.POINTER(ctypes.c_double))
        self.registerType("time_t", ctime)
        self.registerType("time_t*", ctypes.POINTER(ctime))



    def registerType(self, type, value):
        """Register a type against a legal ctypes type"""
        self.registered_types[type] = value
        

    def __parseType(self, type):
        """Convert a prototype definition type from string to a ctypes legal type."""
        type = type.strip()

        if self.registered_types.has_key(type):
            return self.registered_types[type]
        else:
            return getattr(ctypes, type)

        
    def prototype(self, prototype):
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

        match = re.match(self.pattern, prototype)
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

