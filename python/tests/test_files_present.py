import os
import fnmatch

from tests import ResTest


def find_files(path, filter):
    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, filter):
            matches.append(filename)
    return matches


class StdEnKFDebugTest(ResTest):

    def test_all_tests_in_cmakelists(self):
        tests_build = find_files(os.path.join(self.SOURCE_ROOT, "build"), "test_*.py")
        tests_source = find_files(os.path.join(self.SOURCE_ROOT, "python"), "test_*.py")

        diff = set(tests_build).symmetric_difference(set(tests_source))

        assert list(diff) == []
