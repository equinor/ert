import ctypes
from ert.cwrap import CNamespace


class BaseCClass(object):
    namespaces = {}

    def __init__(self, c_pointer, parent=None, is_reference=False):
        self._c_pointer = c_pointer
        self._parent = parent
        self._is_reference = is_reference

        # print("BaseCClass initialized! " + str(self.c_pointer))

    @classmethod
    def cNamespace(cls):
        if not BaseCClass.namespaces.has_key(cls):
            BaseCClass.namespaces[cls] = CNamespace(cls.__name__)
        return BaseCClass.namespaces[cls]

    @classmethod
    def from_param(cls, c_class_object):
        assert isinstance(c_class_object, BaseCClass)

        if c_class_object is None or not hasattr(c_class_object, "_c_pointer"):
            return ctypes.c_void_p()
        else:
            return ctypes.c_void_p(c_class_object._c_pointer)

    @classmethod
    def createPythonObject(cls, c_pointer):
        new_obj = cls.__new__(cls)
        BaseCClass.__init__(new_obj, c_pointer=c_pointer, parent=None, is_reference=False)
        return new_obj

    @classmethod
    def createCReference(cls, c_pointer):
        new_obj = cls.__new__(cls)
        BaseCClass.__init__(new_obj, c_pointer=c_pointer, parent=None, is_reference=True)
        return new_obj

    def setParent(self, parent=None):
        #Todo: should object types also be able to have parents?
        if self._is_reference:
            self._parent = parent
        else:
            raise UserWarning("Can only set parent on reference types!")

    def isReference(self):
        return self._is_reference

    def parent(self):
        return self._parent

    def free(self):
        raise NotImplementedError("A CClass requires a free method!")

    def __del__(self):
        if self.free is not None:
            if not self._is_reference:
                # Important to check the c_pointer; in the case of failed object creation
                # we can have a Python object with c_pointer == None.
                if self._c_pointer:
                    self.free()