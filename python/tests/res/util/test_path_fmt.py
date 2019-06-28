import os
from tests import ResTest
from res.util import PathFormat

class PathFmtTest(ResTest):

    def test_create(self):
        path_fmt = PathFormat("random/path/%d-%d")
        self.assertIn('random/path', repr(path_fmt))
        self.assertTrue(str(path_fmt).startswith('PathFormat('))
