import os
import unittest
from tests import ResTest


class TestFMValidity(ResTest):
    def setUp(self):
        pass

    def _extract_executable(self, filename):
        with open(filename, "r") as f:
            for line in f.readlines():
                l = line.strip().split()
                if len(l) > 1 and l[0] == "EXECUTABLE":
                    return l[1]
        return None

    def _file_exist_and_is_executable(self, file_path):
        return os.path.isfile(file_path) and os.access(file_path, os.X_OK)

    def test_validate_scripts(self):
        fm_path = "share/ert/forward-models"
        fm_path = os.path.join(self.SOURCE_ROOT, fm_path)
        for fm_dir in os.listdir(fm_path):
            fm_dir = os.path.join(fm_path, fm_dir)
            # get all sub-folder in forward-models
            if os.path.isdir(fm_dir):
                files = os.listdir(fm_dir)
                for fn in files:
                    fn = os.path.join(fm_dir, fn)
                    # get all files in sub-folders
                    if os.path.isfile(fn):
                        # extract executable (if any)
                        executable_script = self._extract_executable(fn)
                        if executable_script is not None:
                            self.assertTrue(
                                self._file_exist_and_is_executable(
                                    os.path.join(fm_dir, executable_script)
                                )
                            )


if __name__ == "__main__":
    unittest.main()
