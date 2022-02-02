#  Copyright (C) 2022  Equinor ASA, Norway.
#
#  The file 'analysis_module_options_enum.py' is part of ERT - Ensemble based Reservoir Tool.
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
from cwrap import BaseCEnum


class AnalysisModeEnum(BaseCEnum):
    TYPE_NAME = "analysis_mode_enum"
    ENSEMBLE_SMOOTHER = None
    ITERATED_ENSEMBLE_SMOOTHER = None


AnalysisModeEnum.addEnum("ENSEMBLE_SMOOTHER", 1)
AnalysisModeEnum.addEnum("ITERATED_ENSEMBLE_SMOOTHER", 2)
