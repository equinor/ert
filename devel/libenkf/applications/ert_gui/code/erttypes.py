import datetime
import time
import ctypes

class time_t(ctypes.c_long):

    def __init__(self, value):
        ctypes.c_long.__init__(self, value)

    def time(self):
        return time.localtime(self.value)

    def datetime(self):
        return datetime.date(*self.time()[0:3])

    def __str__(self):
        return "%d %s" % (self.value, str(self.datetime()))

    def __ge__(self, other):
        return self.value >= other.value

    def __lt__(self, other):
        return not self >= other

        
class VectorIterator:
    def __init__(self, data, size):
        self.index = 0
        self.data = data
        self.size = size
    def next(self):
        if self.index == self.size:
            raise StopIteration
        result = self.data[self.index]
        self.index += 1
        return result


class time_vector(ctypes.c_long):
    initialized = False
    lib = None

    def __getitem__(self, item):        
        return self.__class__.lib.time_t_vector_iget(self, item)

    def __iter__(self):
        return VectorIterator(self, self.size())

    def __del__(self):
        self.free()

    def size(self):
        return self.__class__.lib.time_t_vector_size(self)

    def free(self):
        self.__class__.lib.time_t_vector_free(self)

    def getPointer(self):
        return self.__class__.lib.time_t_vector_get_ptr(self)

    @classmethod
    def initialize(cls, ert):
        if not cls.initialized:
            cls.lib = ert.util
            ert.registerType("time_vector", time_vector)
            ert.prototype("int time_t_vector_size(time_vector)", lib=ert.util)
            ert.prototype("time_t time_t_vector_iget(time_vector, int)", lib=ert.util)
            ert.prototype("long time_t_vector_get_ptr(time_vector)", lib=ert.util)
            ert.prototype("void time_t_vector_free(time_vector)", lib=ert.util)

            cls.initialized = True


class double_vector(ctypes.c_long):
    initialized = False
    lib = None

    def __getitem__(self, item):
        return self.__class__.lib.double_vector_iget(self, item)

    def __iter__(self):
        return VectorIterator(self, self.size())

    def __del__(self):
        self.free()

    def size(self):
        return self.__class__.lib.double_vector_size(self)

    def free(self):
        self.__class__.lib.double_vector_free(self)

    def getPointer(self):
        return self.__class__.lib.double_vector_get_ptr(self)

    @classmethod
    def initialize(cls, ert):
        if not cls.initialized:
            cls.lib = ert.util
            ert.registerType("double_vector", double_vector)
            ert.prototype("int double_vector_size(double_vector)", lib=ert.util)
            ert.prototype("double double_vector_iget(double_vector, int)", lib=ert.util)
            ert.prototype("long double_vector_get_ptr(double_vector)", lib=ert.util)
            ert.prototype("void double_vector_free(double_vector)", lib=ert.util)

            cls.initialized = True