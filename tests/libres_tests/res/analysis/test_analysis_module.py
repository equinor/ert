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

import pytest

from res.analysis import AnalysisModule, AnalysisModuleOptionsEnum, AnalysisModeEnum


def test_analysis_module():
    am = AnalysisModule(100, AnalysisModeEnum.ITERATED_ENSEMBLE_SMOOTHER)

    assert am.setVar("ITER", "1")

    assert am.name() == "IES_ENKF"

    assert am.checkOption(AnalysisModuleOptionsEnum.ANALYSIS_ITERABLE)

    assert am.hasVar("ITER")

    assert isinstance(am.getDouble("ENKF_TRUNCATION"), float)

    assert isinstance(am.getInt("ITER"), int)


def test_set_get_var():
    mod = AnalysisModule(100, AnalysisModeEnum.ENSEMBLE_SMOOTHER)

    with pytest.raises(KeyError):
        mod.setVar("NO-NOT_THIS_KEY", 100)

    with pytest.raises(KeyError):
        mod.getInt("NO-NOT_THIS_KEY")
