from ert._c_wrappers.util import PathFormat

from ...libres_utils import ResTest


class PathFmtTest(ResTest):
    def test_create(self):
        path_fmt = PathFormat("random/path/%d-%d")
        self.assertIn("random/path", repr(path_fmt))
        self.assertTrue(str(path_fmt).startswith("PathFormat("))
