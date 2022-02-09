#  Copyright (C) 2013  Equinor ASA, Norway.
#
#  The file 'test_analysis_module.py' is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.


from ecl.util.enums import RngAlgTypeEnum, RngInitModeEnum
from ecl.util.util.rng import RandomNumberGenerator
from libres_utils import ResTest

from res.analysis import AnalysisModule, AnalysisModuleOptionsEnum, AnalysisModeEnum
from res.util import Matrix


class AnalysisModuleTest(ResTest):
    def setUp(self):
        self.rng = RandomNumberGenerator(
            RngAlgTypeEnum.MZRAN, RngInitModeEnum.INIT_DEFAULT
        )

    def test_analysis_module(self):
        am = AnalysisModule(100, AnalysisModeEnum.ITERATED_ENSEMBLE_SMOOTHER)

        self.assertTrue(am.setVar("ITER", "1"))

        self.assertEqual(am.name(), "IES_ENKF")

        self.assertTrue(am.checkOption(AnalysisModuleOptionsEnum.ANALYSIS_ITERABLE))

        self.assertTrue(am.hasVar("ITER"))

        self.assertIsInstance(am.getDouble("ENKF_TRUNCATION"), float)

        self.assertIsInstance(am.getInt("ITER"), int)

    def test_set_get_var(self):
        mod = AnalysisModule(100, AnalysisModeEnum.ENSEMBLE_SMOOTHER)
        with self.assertRaises(KeyError):
            mod.setVar("NO-NOT_THIS_KEY", 100)

        with self.assertRaises(KeyError):
            mod.getInt("NO-NOT_THIS_KEY")

    def construct_matrix(self, n, vals):
        """Constructs n*n matrix with vals as entries"""
        self.assertEqual(n * n, len(vals))
        m = Matrix(n, n)
        idx = 0
        for i in range(n):
            for j in range(n):
                m[(i, j)] = vals[idx]
                idx += 1
        return m

    def _n_identity_mcs(self, n=6, s=3):
        """return n copies of the identity matrix on s*s elts"""
        return tuple([Matrix.identity(s) for i in range(n)])

    def _matrix_close(self, m1, m2, epsilon=0.01):
        """Check that matrices m1 and m2 are of same dimension and that they are
        pointwise within epsilon difference."""

        c = m1.columns()
        r = m1.rows()
        self.assertEqual(c, m2.columns(), "Number of columns for m1 differ from m2")
        self.assertEqual(r, m2.rows(), "Number of rows for m1 differ from m2")
        for i in range(0, c):
            for j in range(0, r):
                pos = (i, j)
                diff = abs(m1[pos] - m2[pos])
                self.assertTrue(
                    diff <= epsilon,
                    "Matrices differ at (i,j) = (%d,%d). %f != %f"
                    % (i, j, m1[pos], m2[pos]),
                )
