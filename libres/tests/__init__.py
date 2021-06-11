from pathlib import Path
import types
import os.path
import functools
from ecl.util.test import ExtendedTestCase


def libres_source_root() -> Path:
    src = Path("@CMAKE_CURRENT_SOURCE_DIR@/../..")
    if src.is_dir():
        return src.relative_to(Path.cwd())

    # If the file was not correctly configured by cmake, look for the source
    # folder, assuming the build folder is inside the source folder.
    test_path = Path(__file__)
    while test_path != Path("/"):
        if (test_path / ".git").is_dir():
            return test_path / "libres"
        test_path = test_path.parent
    raise RuntimeError("Cannot find the source folder")


class ResTest(ExtendedTestCase):
    SOURCE_ROOT = libres_source_root()
    TESTDATA_ROOT = SOURCE_ROOT / ".." / "test-data"
    SHARE_ROOT = SOURCE_ROOT.parent / "share"
    EQUINOR_DATA = (TESTDATA_ROOT / "Equinor").is_symlink()

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
