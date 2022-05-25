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

from res.analysis import AnalysisModule


def test_analysis_module():
    am = AnalysisModule(2)

    assert am.setVar("ENKF_TRUNCATION", "1.0")

    assert am.name() == "IES_ENKF"

    assert am.hasVar("IES_INVERSION")

    assert isinstance(am.getDouble("ENKF_TRUNCATION"), float)


def test_set_get_var():
    mod = AnalysisModule(1)

    with pytest.raises(KeyError):
        mod.setVar("NO-NOT_THIS_KEY", 100)

    with pytest.raises(KeyError):
        mod.getInt("NO-NOT_THIS_KEY")
