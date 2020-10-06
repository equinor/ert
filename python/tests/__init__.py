import types
import os.path
import functools
from ecl.util.test import ExtendedTestCase


def source_root():
    src = "@CMAKE_CURRENT_SOURCE_DIR@/../.."
    if os.path.isdir(src):
        return os.path.realpath(src)

    # If the file was not correctly configured by cmake, look for the source
    # folder, assuming the build folder is inside the source folder.
    path_list = os.path.dirname(os.path.abspath(__file__)).split("/")
    while len(path_list) > 0:
        git_path = os.path.join(os.sep, "/".join(path_list), ".git")
        if os.path.isdir(git_path):
            return os.path.join(os.sep, *path_list)
        path_list.pop()
    raise RuntimeError("Cannot find the source folder")


class ResTest(ExtendedTestCase):
    SOURCE_ROOT = source_root()
    TESTDATA_ROOT = os.path.join(SOURCE_ROOT, "test-data")
    SHARE_ROOT = os.path.join(SOURCE_ROOT, "share")
    EQUINOR_DATA = os.path.islink(os.path.join(TESTDATA_ROOT, "Equinor"))

    def assertItemsEqual(self, data1, data2):
        if len(data1) != len(data2):
            raise AssertionError("Element count not equal.")

        for value in data1:
            if not value in data2:
                raise AssertionError(value, "not in", data2)

    @classmethod
    def createSharePath(cls, path):
        if cls.SHARE_ROOT is None:
            raise Exception(
                "Trying to create directory rooted in 'SHARE_ROOT' - variable 'SHARE_ROOT' is not set."
            )
        return os.path.realpath(os.path.join(cls.SHARE_ROOT, path))
