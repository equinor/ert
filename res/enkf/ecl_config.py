#  Copyright (C) 2012  Equinor ASA, Norway.
#
#  The file 'ecl_config.py' is part of ERT - Ensemble based Reservoir Tool.
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

from cwrap import BaseCClass
from res import ResPrototype
from ecl.grid import EclGrid
from ecl.summary import EclSum
from ecl.util.util import StringList, CTime
from res.util import UIReturn
from res.enkf import ConfigKeys
import os
from datetime import datetime


class EclConfig(BaseCClass):
    TYPE_NAME = "ecl_config"

    _alloc = ResPrototype("void* ecl_config_alloc(config_content)", bind=False)
    _alloc_full = ResPrototype(
        "void* ecl_config_alloc_full(  bool, \
                                                                            char*, \
                                                                            ecl_grid, \
                                                                            char*, \
                                                                            stringlist, \
                                                                            time_t, \
                                                                            char*)",
        bind=False,
    )
    _free = ResPrototype("void  ecl_config_free( ecl_config )")
    _get_data_file = ResPrototype("char* ecl_config_get_data_file(ecl_config)")
    _set_data_file = ResPrototype("void  ecl_config_set_data_file(ecl_config , char*)")
    _validate_data_file = ResPrototype(
        "ui_return_obj ecl_config_validate_data_file(ecl_config , char*)"
    )
    _get_gridfile = ResPrototype("char* ecl_config_get_gridfile(ecl_config)")
    _set_gridfile = ResPrototype("void  ecl_config_set_grid(ecl_config, char*)")
    _validate_gridfile = ResPrototype(
        "ui_return_obj ecl_config_validate_grid(ecl_config, char*)"
    )
    _get_grid = ResPrototype("ecl_grid_ref ecl_config_get_grid(ecl_config)")
    _get_refcase_name = ResPrototype("char* ecl_config_get_refcase_name(ecl_config)")
    _get_refcase = ResPrototype("ecl_sum_ref ecl_config_get_refcase(ecl_config)")
    _load_refcase = ResPrototype("void  ecl_config_load_refcase(ecl_config, char*)")
    _validate_refcase = ResPrototype(
        "ui_return_obj ecl_config_validate_refcase(ecl_config, char*)"
    )
    _has_refcase = ResPrototype("bool  ecl_config_has_refcase(ecl_config)")
    _get_depth_unit = ResPrototype("char* ecl_config_get_depth_unit(ecl_config)")
    _get_pressure_unit = ResPrototype("char* ecl_config_get_pressure_unit(ecl_config)")
    _active = ResPrototype("bool ecl_config_active(ecl_config)")
    _get_last_history_restart = ResPrototype(
        "int ecl_config_get_last_history_restart(ecl_config)"
    )
    _get_end_date = ResPrototype("time_t ecl_config_get_end_date(ecl_config)")
    _get_num_cpu = ResPrototype("int ecl_config_get_num_cpu(ecl_config)")

    def __init__(self, config_content=None, config_dict=None):

        if config_content is not None and config_dict is not None:
            raise ValueError(
                "Error: EclConfig can not be instantiated with multiple config objects"
            )
        c_ptr = None
        if config_dict is None:
            c_ptr = self._alloc(config_content)

        if config_dict is not None:
            # ECLBASE_KEY
            have_eclbase = config_dict.get(ConfigKeys.ECLBASE) is not None

            # DATA_FILE_KEY
            data_file = config_dict.get(ConfigKeys.DATA_FILE)
            if data_file is not None:
                data_file = os.path.realpath(data_file)
                if not os.path.isfile(data_file):
                    raise ValueError("Error: data file is not a file")

            # GRID_KEY
            grid = None
            grid_file = config_dict.get(ConfigKeys.GRID)
            if grid_file is not None:
                grid_file = os.path.realpath(grid_file)
                if not os.path.isfile(grid_file):
                    raise ValueError("Error: grid file is not a file")
                grid = EclGrid.load_from_file(grid_file)

            # REFCASE_KEY
            refcase_default = config_dict.get(ConfigKeys.REFCASE)
            if refcase_default is not None:
                refcase_default = os.path.realpath(refcase_default)

            # REFCASE_LIST_KEY
            refcase_list = StringList()
            for refcase in config_dict.get(ConfigKeys.REFCASE_LIST, []):
                refcase_list.append(refcase)

            # END_DATE_KEY
            end_date = CTime(
                datetime.strptime(
                    config_dict.get(ConfigKeys.END_DATE, "31/12/1969"), "%d/%m/%Y"
                )
            )

            # SCHEDULE_PREDICTION_FILE_KEY
            schedule_prediction_file = config_dict.get(
                ConfigKeys.SCHEDULE_PREDICTION_FILE
            )

            c_ptr = self._alloc_full(
                have_eclbase,
                data_file,
                grid,
                refcase_default,
                refcase_list,
                end_date,
                schedule_prediction_file,
            )
            if grid is not None:
                grid.convertToCReference(None)

        if c_ptr:
            super(EclConfig, self).__init__(c_ptr)
        else:
            raise RuntimeError("Internal error: Failed constructing EclConfig!")

    def free(self):
        self._free()

    def getDataFile(self):
        return self._get_data_file()

    def setDataFile(self, datafile):
        self._set_data_file(datafile)

    def validateDataFile(self, datafile):
        """@rtype: UIReturn"""
        return self._validate_data_file(datafile)

    # -----------------------------------------------------------------

    def get_gridfile(self):
        """@rtype: str"""
        return self._get_gridfile()

    def set_gridfile(self, gridfile):
        self._set_gridfile(gridfile)

    def validateGridFile(self, gridfile):
        return self._validate_gridfile(gridfile)

    def getGrid(self):
        return self._get_grid()

    def getRefcaseName(self):
        return self._get_refcase_name()

    def loadRefcase(self, refcase):
        self._load_refcase(refcase)

    def getRefcase(self):
        """@rtype: EclSum"""
        refcase = self._get_refcase()
        if not refcase is None:
            refcase.setParent(self)

        return refcase

    def validateRefcase(self, refcase):
        return self._validate_refcase(refcase)

    def hasRefcase(self):
        """@rtype: bool"""
        return self._has_refcase()

    # -----------------------------------------------------------------

    def getDepthUnit(self):
        return self._get_depth_unit()

    def getPressureUnit(self):
        return self._get_pressure_unit()

    # -----------------------------------------------------------------

    def getEndDate(self):
        return self._get_end_date()

    def active(self):
        """
        Has ECLIPSE been configured?"
        """
        return self._active()

    def getLastHistoryRestart(self):
        return self._get_last_history_restart()

    @property
    def num_cpu(self):
        """
        Returns numbers cpu to be used defined in a DATA file
        """
        return self._get_num_cpu()

    def __eq__(self, other):
        if self.getDataFile() != other.getDataFile():
            return False

        if self.get_gridfile() != other.get_gridfile():
            return False

        if self.getRefcaseName() != other.getRefcaseName():
            return False

        if self.getDepthUnit() != other.getDepthUnit():
            return False

        if self.getPressureUnit() != other.getPressureUnit():
            return False

        if self.getEndDate() != other.getEndDate():
            return False

        return True
