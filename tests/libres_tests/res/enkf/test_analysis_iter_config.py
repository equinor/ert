#  Copyright (C) 2014  Equinor ASA, Norway.
#
#  The file 'test_analysis_iter_config.py' is part of ERT
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

from ert._c_wrappers.enkf import AnalysisIterConfig


def test_set_analysis_iter_config():
    c = AnalysisIterConfig()
    assert repr(c).startswith("AnalysisIterConfig")

    assert not c.caseFormatSet()
    c.setCaseFormat("case%d")
    assert c.caseFormatSet()

    assert not c.numIterationsSet()
    c.setNumIterations(1)
    assert c.numIterationsSet()


def test_analysis_iter_config_constructor():
    config_dict = {
        "ITER_CASE": "ITERATED_ENSEMBLE_SMOOTHER%d",
        "ITER_COUNT": 4,
        "ITER_RETRY_COUNT": 4,
    }
    c_default = AnalysisIterConfig()
    c_dict = AnalysisIterConfig(config_dict=config_dict)
    assert c_default == c_dict
