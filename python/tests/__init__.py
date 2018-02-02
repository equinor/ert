import os.path
from ecl.util.test import ExtendedTestCase


def source_root():
    path_list = os.path.dirname(os.path.abspath(__file__)).split("/")
    while True:
        git_path = os.path.join(os.sep, "/".join(path_list), ".git")
        if os.path.isdir(git_path):
            return os.path.join(os.sep, *path_list)
        path_list.pop()


class ErtTest(ExtendedTestCase):
    SOURCE_ROOT = source_root()
    TESTDATA_ROOT = os.path.join(SOURCE_ROOT, "test-data")
    SHARE_ROOT = os.path.join(SOURCE_ROOT,"share")


    @classmethod
    def createSharePath(cls, path):
        if cls.SHARE_ROOT is None:
            raise Exception("Trying to create directory rooted in 'SHARE_ROOT' - variable 'SHARE_ROOT' is not set.")
        return os.path.realpath(os.path.join(cls.SHARE_ROOT , path))
