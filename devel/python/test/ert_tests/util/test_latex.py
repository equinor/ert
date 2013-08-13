import os
from unittest2 import TestCase, skip
from ert.util import LaTeX

local_path = "test-data/local/util/latex"
statoil_path = "test-data/Statoil/util/latex"

class TestLatex(TestCase):
    def test_simple_compile(self):
        lx = LaTeX("%s/test_OK.tex" % local_path)
        self.assertTrue(lx.compile())

        lx = LaTeX("%s/test_error.tex" % local_path)
        self.assertFalse(lx.compile())

    @skip("Unknown errors!")
    def test_cleanup( self ):
        lx = LaTeX("%s/report_OK.tex" % statoil_path, in_place=True)
        self.assertTrue(lx.in_place)
        self.assertTrue(lx.compile())
        for ext in ["log", "aux", "nav", "out", "snm", "toc"]:
            self.assertFalse(os.path.exists("%s/report_OK.%s" % (statoil_path, ext)))

        lx = LaTeX("%s/report_OK.tex" % statoil_path, in_place=False)
        self.assertFalse(lx.in_place)
        run_path = lx.runpath
        self.assertTrue(lx.compile())
        self.assertFalse(os.path.exists(run_path))

        lx = LaTeX("%s/report_OK.tex" % statoil_path, in_place=False)
        run_path = lx.runpath
        self.assertTrue(lx.compile(cleanup=False))
        self.assertTrue(os.path.exists("%s/report_OK.log" % run_path))


    @skip("Unknown errors!")
    def test_report(self):
        lx = LaTeX("%s/report_error.tex" % statoil_path)
        lx.timeout = 4
        self.assertFalse(lx.compile())

        lx = LaTeX("%s/report_OK.tex" % statoil_path)
        self.assertTrue(lx.compile())

    @skip("Unknown errors!")
    def test_target(self):
        lx = LaTeX("%s/report_OK.tex" % statoil_path)
        self.assertTrue(lx.compile())
        self.assertTrue(os.path.exists(lx.target))