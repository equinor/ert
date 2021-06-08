#  Copyright (C) 2013  Equinor ASA, Norway.
#
#  The file 'job_status_type_enum.py' is part of ERT - Ensemble based Reservoir Tool.
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


class RunStatusType(BaseCEnum):
    TYPE_NAME = "run_status_type_enum"

    JOB_LOAD_FAILURE = None
    JOB_RUN_FAILURE = None

    @classmethod
    def from_string(cls, string):
        pass


RunStatusType.addEnum("JOB_RUN_FAILURE", 2)
RunStatusType.addEnum("JOB_LOAD_FAILURE", 3)
