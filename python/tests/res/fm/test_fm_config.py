import os
import unittest
from tests import ResTest

class TestFMValidity(ResTest):

    def setUp(self):
        pass

    def _extract_executable(self,filename):
        executable_list = []
        with open(filename,'r') as f:
            for line in f.readlines():
                l = [a.strip() for a in line.split(' ') if a not in ['', ' ', '  ']]
                if l[0]=='EXECUTABLE':
                    executable_list.append(l[1])
        return executable_list

    def _file_exist_and_is_executable(self,file_path):
        return os.path.isfile(file_path) and os.access(file_path, os.X_OK)

    def test_validate_scripts(self):
        fm_path = ['share/ert/forward-models/res',
                   'share/ert/forward-models/shell',
                   'share/ert/forward-models/templating']
        for fdir in fm_path:
            fpath = os.path.join(self.SOURCE_ROOT,fdir)
            files = os.listdir(fpath)
            for fn in files:
                fn = os.path.join(fpath,fn)
                if os.path.isfile(fn):
                    executable_script = self._extract_executable(os.path.join(fdir,fn))
                    valid_executable = [self._file_exist_and_is_executable(os.path.join(fpath,es))
                                        for es in executable_script]
                    self.assertFalse(False in valid_executable)


if __name__ == "__main__":
    unittest.main()
