#  Copyright (C) 2017  Equinor ASA, Norway.
#
#  The file '__init__.py' is part of ERT - Ensemble based Reservoir Tool.
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
from cwrap import Prototype
import res

from .substitution_list import SubstitutionList
from .enums import LLSQResultEnum
from .log import Log
from .res_version import ResVersion
from .res_log import ResLog
from .ui_return import UIReturn
from .path_format import PathFormat
from .matrix import Matrix
from .stat import polyfit
