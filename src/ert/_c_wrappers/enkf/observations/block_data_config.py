import ctypes

from cwrap import BaseCClass

from ert._c_wrappers.enkf.config import FieldConfig


class BlockDataConfig(BaseCClass):
    TYPE_NAME = "block_data_config"

    def __init__(self):
        raise NotImplementedError("Cannot instantiate BlockDataConfig!")

    @classmethod
    def from_param(cls, c_class_object):
        if c_class_object is None:
            return ctypes.c_void_p()
        elif isinstance(c_class_object, FieldConfig):
            return FieldConfig.from_param(c_class_object)

        else:
            raise ValueError("Currently ONLY field data is supported")

    def free(self):
        pass
